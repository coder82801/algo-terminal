import gc
import math
import os
import time
import traceback
from datetime import datetime
from typing import Dict, List, Tuple

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(page_title="NextDay Scanner Pro", layout="wide")
st.title("🎯 NextDay Scanner Pro (Top 10 + En İyi 3)")

DEBUG_MODE = st.sidebar.checkbox("Debug Mode", value=False)

env_api_key = os.getenv("ALPACA_API_KEY", "")
env_secret_key = os.getenv("ALPACA_SECRET_KEY", "")

st.sidebar.header("Alpaca API (Paper)")
api_key = st.sidebar.text_input("API Key ID", value=env_api_key, type="password")
secret_key = st.sidebar.text_input("Secret Key", value=env_secret_key, type="password")

TV_URL = "https://scanner.tradingview.com/america/scan"
TV_HEADERS = {"User-Agent": "Mozilla/5.0"}
YF_MAX_CHUNK = 120
YF_SLEEP_SEC = 1.0
YF_RETRIES = 2

# Daha çok hisse odaklı sonuç için ETF/ETN/kapalı uçlu fon benzeri yaygın işaretler.
NON_STOCK_HINTS = (
    " ETF", " ETN", " TRUST", " FUND", " INCOME", " BOND", " TREASURY", " ULTRA", " BEAR", " BULL"
)


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================
def dbg(msg: str) -> None:
    if DEBUG_MODE:
        st.sidebar.write(msg)


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def format_price(x: float) -> str:
    if pd.isna(x):
        return "-"
    if x < 1:
        return f"{x:.4f}"
    if x < 10:
        return f"{x:.3f}"
    return f"{x:.2f}"


def get_api(api_key_value: str, secret_key_value: str):
    return tradeapi.REST(
        key_id=api_key_value,
        secret_key=secret_key_value,
        base_url="https://paper-api.alpaca.markets",
        api_version="v2",
    )


def normalize_for_yf(symbol: str) -> str:
    """Yahoo Finance için noktalı class hisseleri tireye çevirir."""
    return symbol.replace(".", "-").strip().upper()


def denormalize_from_yf(symbol: str) -> str:
    return symbol.replace("-", ".").strip().upper()


def calc_true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = calc_true_range(df)
    return tr.rolling(period).mean()


def calc_session_vwap(intraday_df: pd.DataFrame) -> float:
    if intraday_df.empty or "Volume" not in intraday_df.columns:
        return np.nan
    temp = intraday_df.copy().dropna(subset=["High", "Low", "Close", "Volume"])
    if temp.empty:
        return np.nan
    temp["Typical_Price"] = (temp["High"] + temp["Low"] + temp["Close"]) / 3
    temp["VP"] = temp["Typical_Price"] * temp["Volume"]
    vol_sum = temp["Volume"].cumsum()
    vp_sum = temp["VP"].cumsum()
    if vol_sum.iloc[-1] == 0:
        return np.nan
    return float((vp_sum / vol_sum).iloc[-1])


def calc_closing_strength(last_close: float, last_low: float, last_high: float) -> float:
    day_range = last_high - last_low
    if day_range <= 0:
        return np.nan
    return (last_close - last_low) / day_range


def dynamic_stop_limit(stop_price: float) -> float:
    if stop_price <= 0:
        return stop_price
    if stop_price < 1:
        offset = max(0.005, stop_price * 0.01)
    elif stop_price < 5:
        offset = max(0.02, stop_price * 0.0075)
    else:
        offset = max(0.05, stop_price * 0.005)
    return round(max(0.01, stop_price - offset), 4)


def chunked(items: List[str], size: int) -> List[List[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def tv_scan(filters: List[dict], columns: List[str], sort_field: str, max_records: int, page_size: int = 100) -> List[dict]:
    rows: List[dict] = []
    start = 0
    while start < max_records:
        payload = {
            "filter": filters,
            "options": {"lang": "en"},
            "markets": ["america"],
            "symbols": {"query": {"types": ["stock"]}, "tickers": []},
            "columns": columns,
            "sort": {"sortBy": sort_field, "sortOrder": "desc"},
            "range": [start, start + page_size - 1],
        }
        try:
            res = requests.post(TV_URL, json=payload, headers=TV_HEADERS, timeout=20)
            res.raise_for_status()
            data = res.json().get("data", [])
        except Exception as exc:
            dbg(f"TradingView scan error [{start}]: {exc}")
            break
        if not data:
            break
        rows.extend(data)
        if len(data) < page_size:
            break
        start += page_size
    return rows


@st.cache_data(ttl=10)
def get_intraday_gainers(session: str) -> pd.DataFrame:
    try:
        if "Pre-Market" in session:
            sort_field, vol_field, price_field, min_vol = "premarket_change", "premarket_volume", "premarket_close", 10000
        elif "After-Hours" in session:
            sort_field, vol_field, price_field, min_vol = "postmarket_change", "postmarket_volume", "postmarket_close", 10000
        else:
            sort_field, vol_field, price_field, min_vol = "change", "volume", "close", 50000

        filters = [
            {"left": vol_field, "operation": "greater", "right": min_vol},
            {"left": price_field, "operation": "greater", "right": 0.50},
            {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
        ]
        rows = tv_scan(
            filters=filters,
            columns=["name", "description", price_field, sort_field, vol_field],
            sort_field=sort_field,
            max_records=25,
            page_size=25,
        )
        out = []
        for row in rows[:15]:
            try:
                d = row["d"]
                price = safe_float(d[2], 0.0)
                out.append(
                    {
                        "Hisse": d[0],
                        "Şirket": d[1],
                        "Fiyat ($)": round(price, 4) if price < 1 else round(price, 2),
                        "Artış (%)": round(safe_float(d[3], 0.0), 2),
                        "Hacim": f"{int(safe_float(d[4], 0.0)):,}",
                    }
                )
            except Exception as exc:
                dbg(f"Intraday row parse error: {exc}")
        return pd.DataFrame(out)
    except Exception as exc:
        dbg(f"get_intraday_gainers error: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=900)
def fetch_tv_candidates(scan_mode: str, max_records: int) -> pd.DataFrame:
    base_filters = [
        {"left": "close", "operation": "greater", "right": 2.0},
        {"left": "volume", "operation": "greater", "right": 250000},
        {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
    ]

    if scan_mode.startswith("A"):
        algo_filters = [{"left": "relative_volume_10d_calc", "operation": "greater", "right": 1.5}]
        sort_field = "relative_volume_10d_calc"
    elif scan_mode.startswith("B"):
        algo_filters = [
            {"left": "gap", "operation": "greater", "right": 2.0},
            {"left": "relative_volume_10d_calc", "operation": "greater", "right": 2.0},
        ]
        sort_field = "gap"
    else:
        algo_filters = []
        sort_field = "relative_volume_10d_calc"

    columns = [
        "name", "description", "close", "volume", "relative_volume_10d_calc", "gap", "market_cap_basic"
    ]
    rows = tv_scan(base_filters + algo_filters, columns, sort_field, max_records=max_records, page_size=100)

    parsed = []
    seen = set()
    for row in rows:
        try:
            d = row["d"]
            symbol = str(d[0]).upper().strip()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            description = str(d[1]) if d[1] is not None else ""
            close = safe_float(d[2], np.nan)
            volume = safe_float(d[3], np.nan)
            rvol = safe_float(d[4], np.nan)
            gap = safe_float(d[5], np.nan)
            mcap = safe_float(d[6], np.nan)
            parsed.append(
                {
                    "Symbol": symbol,
                    "Description": description,
                    "CloseTV": close,
                    "VolumeTV": volume,
                    "RVOLTV": rvol,
                    "GapTV": gap,
                    "MarketCapTV": mcap,
                }
            )
        except Exception as exc:
            dbg(f"TV parse error: {exc}")
    return pd.DataFrame(parsed)


def likely_non_stock(description: str, symbol: str) -> bool:
    desc = (description or "").upper()
    if any(hint in desc for hint in NON_STOCK_HINTS):
        return True
    # Leveraged/inverse/covered-call ETF benzeri sık görülen yapılar için kaba koruma.
    if symbol.endswith(("Y", "X")) and any(token in desc for token in ["INCOME", "COVERED", "TREASURY"]):
        return True
    return False


def download_yf_history(tickers: List[str], period: str = "260d") -> Tuple[Dict[str, pd.DataFrame], List[dict]]:
    """
    yfinance verisini chunk'lar halinde indirir; rate limit ve sembol hatalarını loglar.
    Dönüş: {orijinal_symbol: OHLCV DataFrame}, failed_logs
    """
    symbol_map = {normalize_for_yf(t): t for t in tickers}
    normalized = list(symbol_map.keys())
    history_map: Dict[str, pd.DataFrame] = {}
    failed: List[dict] = []

    for batch in chunked(normalized, YF_MAX_CHUNK):
        batch_ok = False
        last_exc = None
        for attempt in range(YF_RETRIES + 1):
            try:
                raw = yf.download(
                    tickers=batch,
                    period=period,
                    progress=False,
                    threads=False,
                    auto_adjust=False,
                    group_by="ticker",
                )
                batch_ok = True
                last_exc = None
                # MultiIndex veya tek kolonlu durumları ayır.
                for yf_symbol in batch:
                    original = symbol_map[yf_symbol]
                    try:
                        if isinstance(raw.columns, pd.MultiIndex):
                            if yf_symbol not in raw.columns.get_level_values(0):
                                failed.append({"Hisse": original, "Neden": "Yahoo veri dönmedi"})
                                continue
                            sub = raw[yf_symbol].copy()
                        else:
                            # Tek sembollü batch senaryosu.
                            sub = raw.copy()
                        need_cols = ["Open", "High", "Low", "Close", "Volume"]
                        if not all(c in sub.columns for c in need_cols):
                            failed.append({"Hisse": original, "Neden": "Eksik OHLCV kolonları"})
                            continue
                        sub = sub[need_cols].dropna()
                        if sub.empty:
                            failed.append({"Hisse": original, "Neden": "Boş fiyat geçmişi"})
                            continue
                        history_map[original] = sub
                    except Exception as exc:
                        failed.append({"Hisse": original, "Neden": f"Parse hatası: {exc}"})
                break
            except Exception as exc:
                last_exc = exc
                sleep_time = YF_SLEEP_SEC * (attempt + 1) * 2
                time.sleep(sleep_time)
        if not batch_ok:
            for yf_symbol in batch:
                original = symbol_map[yf_symbol]
                failed.append({"Hisse": original, "Neden": f"İndirme hatası: {last_exc}"})
        time.sleep(YF_SLEEP_SEC)
    return history_map, failed


@st.cache_data(ttl=300)
def get_intraday_vwap_for_ticker(symbol: str) -> float:
    try:
        yf_symbol = normalize_for_yf(symbol)
        intraday = yf.Ticker(yf_symbol).history(period="1d", interval="5m", prepost=True, auto_adjust=False)
        if intraday.empty:
            return np.nan
        return calc_session_vwap(intraday)
    except Exception as exc:
        dbg(f"Intraday VWAP error for {symbol}: {exc}")
        return np.nan


def compute_market_context(spy_df: pd.DataFrame) -> dict:
    if spy_df is None or spy_df.empty or len(spy_df) < 20:
        return {"spy_ret_10d": 0.0, "spy_ret_20d": 0.0, "market_bias": "neutral"}
    spy_ret_10d = (spy_df["Close"].iloc[-1] - spy_df["Close"].iloc[-10]) / spy_df["Close"].iloc[-10]
    spy_ret_20d = (spy_df["Close"].iloc[-1] - spy_df["Close"].iloc[-20]) / spy_df["Close"].iloc[-20]
    market_bias = "bull" if spy_ret_10d > 0 and spy_ret_20d > 0 else "bear" if spy_ret_10d < 0 and spy_ret_20d < 0 else "neutral"
    return {"spy_ret_10d": spy_ret_10d, "spy_ret_20d": spy_ret_20d, "market_bias": market_bias}


def evaluate_symbol(
    symbol: str,
    desc: str,
    df: pd.DataFrame,
    market_ctx: dict,
    scan_mode: str,
) -> dict | None:
    if df is None or df.empty or len(df) < 220:
        return None

    last_close = float(df["Close"].iloc[-1])
    last_open = float(df["Open"].iloc[-1])
    last_high = float(df["High"].iloc[-1])
    last_low = float(df["Low"].iloc[-1])
    last_volume = float(df["Volume"].iloc[-1])

    df = df.copy()
    df["ATR14"] = calc_atr(df, 14)
    df["ATR5"] = calc_atr(df, 5)
    df["ATR20"] = calc_atr(df, 20)
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    atr14 = safe_float(df["ATR14"].iloc[-1])
    atr5 = safe_float(df["ATR5"].iloc[-1])
    atr20 = safe_float(df["ATR20"].iloc[-1])
    sma50 = safe_float(df["SMA50"].iloc[-1])
    sma200 = safe_float(df["SMA200"].iloc[-1])

    if pd.isna(atr14) or atr14 <= 0:
        return None

    close_strength = calc_closing_strength(last_close, last_low, last_high)
    avg_vol20 = df["Volume"].rolling(20).mean().iloc[-1]
    rvol = last_volume / avg_vol20 if avg_vol20 and avg_vol20 > 0 else np.nan

    prev_close = float(df["Close"].iloc[-2])
    gap_pct = ((last_open - prev_close) / prev_close) * 100 if prev_close > 0 else np.nan

    prior_20d_high = df["High"].shift(1).rolling(20).max().iloc[-1]
    dist_to_20d_high = ((prior_20d_high - last_close) / last_close) * 100 if pd.notna(prior_20d_high) and prior_20d_high > last_close else 0.0

    stock_ret_10d = (df["Close"].iloc[-1] - df["Close"].iloc[-10]) / df["Close"].iloc[-10]
    rs_10d_spy = stock_ret_10d - market_ctx["spy_ret_10d"]

    # OBV eğimi
    delta = df["Close"].diff().fillna(0)
    obv = np.where(delta > 0, df["Volume"], np.where(delta < 0, -df["Volume"], 0))
    df["OBV"] = pd.Series(obv, index=df.index).cumsum()
    obv_slope_10 = float(df["OBV"].iloc[-1] - df["OBV"].iloc[-10])

    up_vol = df.loc[df["Close"] > df["Open"], "Volume"].tail(15).mean()
    down_vol = df.loc[df["Close"] < df["Open"], "Volume"].tail(15).mean()

    intraday_vwap = get_intraday_vwap_for_ticker(symbol)
    if pd.isna(intraday_vwap):
        intraday_vwap = (last_high + last_low + last_close) / 3
    above_vwap = last_close > intraday_vwap

    # ---------- Top 10 aday skoru ----------
    score = 0
    notes = []

    if pd.notna(rvol):
        if rvol >= 4:
            score += 26
            notes.append("RVOL 4x+")
        elif rvol >= 3:
            score += 22
            notes.append("RVOL 3x+")
        elif rvol >= 2:
            score += 16
            notes.append("RVOL 2x+")
        elif rvol >= 1.5:
            score += 10
            notes.append("RVOL 1.5x+")

    if pd.notna(close_strength):
        if close_strength >= 0.90:
            score += 18
            notes.append("Tepe kapanış")
        elif close_strength >= 0.75:
            score += 14
            notes.append("Güçlü kapanış")
        elif close_strength >= 0.60:
            score += 8

    if above_vwap:
        score += 12
        notes.append("VWAP üstü")
    else:
        score -= 6

    if rs_10d_spy > 0.08:
        score += 14
        notes.append("RS çok güçlü")
    elif rs_10d_spy > 0.03:
        score += 9
        notes.append("RS güçlü")
    elif rs_10d_spy > 0:
        score += 5

    if pd.notna(atr5) and pd.notna(atr20):
        if atr5 < atr20 * 0.85:
            score += 8
            notes.append("Sıkışma")
        elif atr5 > atr20 * 1.15:
            score += 4
            notes.append("ATR genişliyor")

    if dist_to_20d_high <= 2:
        score += 12
        notes.append("20G kırılıma çok yakın")
    elif dist_to_20d_high <= 5:
        score += 6

    if pd.notna(sma200) and last_close > sma200:
        score += 8
        notes.append("200MA üstü")

    if obv_slope_10 > 0:
        score += 6
        notes.append("OBV pozitif")

    if market_ctx["market_bias"] == "bear":
        score -= 6

    # ---------- Kategori ----------
    category = None
    if scan_mode.startswith("A"):
        if (
            pd.notna(rvol) and rvol >= 1.5
            and pd.notna(close_strength) and close_strength >= 0.75
            and above_vwap
            and rs_10d_spy > 0
            and dist_to_20d_high <= 2
        ):
            category = "Breakout"
    elif scan_mode.startswith("B"):
        if (
            pd.notna(gap_pct) and gap_pct >= 2.0
            and pd.notna(rvol) and rvol >= 2.0
            and above_vwap
            and pd.notna(close_strength) and close_strength >= 0.70
        ):
            category = "Continuation"
    else:
        if (
            pd.notna(sma200) and last_close > sma200
            and pd.notna(up_vol) and pd.notna(down_vol) and up_vol > down_vol * 1.2
            and pd.notna(atr5) and pd.notna(atr20) and atr5 < atr20
            and obv_slope_10 > 0
        ):
            category = "Accumulation"

    if category is None:
        return None

    # ---------- Top 3 için ikinci skor ----------
    rvol_score = min((rvol / 3.0), 1.0) if pd.notna(rvol) else 0.0
    close_strength_score = float(np.clip(close_strength, 0, 1)) if pd.notna(close_strength) else 0.0
    vwap_score = 1.0 if above_vwap else 0.0
    breakout_score = 1 - min(dist_to_20d_high / 3.0, 1.0)
    compression_score = max(0.0, 1 - (atr5 / atr20)) if pd.notna(atr5) and pd.notna(atr20) and atr20 > 0 else 0.0
    rs_score = float(np.clip(rs_10d_spy / 0.10, 0, 1))
    clean_chart_score = 1.0 if close_strength_score >= 0.70 else 0.5 if close_strength_score >= 0.55 else 0.0

    final_top3_score = (
        0.25 * rvol_score
        + 0.20 * close_strength_score
        + 0.15 * breakout_score
        + 0.15 * compression_score
        + 0.10 * vwap_score
        + 0.10 * rs_score
        + 0.05 * clean_chart_score
    )

    # ---------- Giriş / Stop / Hedef ----------
    entry_vwap_pullback = intraday_vwap
    entry_breakout_retest = prior_20d_high if pd.notna(prior_20d_high) else last_close
    preferred_entry = entry_vwap_pullback if above_vwap else entry_breakout_retest
    stop_price = preferred_entry - (1.2 * atr14)
    stop_limit_price = dynamic_stop_limit(stop_price)
    risk_per_share = max(preferred_entry - stop_price, 0.01)
    tp1 = preferred_entry + (1.0 * risk_per_share)
    tp2 = preferred_entry + (2.0 * risk_per_share)
    tp15 = preferred_entry * 1.15

    return {
        "Symbol": symbol,
        "Description": desc,
        "Category": category,
        "Close": round(last_close, 4 if last_close < 1 else 2),
        "RVOL": round(rvol, 2) if pd.notna(rvol) else np.nan,
        "Close_Strength": round(close_strength, 2) if pd.notna(close_strength) else np.nan,
        "Dist_to_20D_High_%": round(dist_to_20d_high, 2),
        "VWAP": round(intraday_vwap, 4 if intraday_vwap < 1 else 2),
        "RS_10D_vs_SPY_%": round(rs_10d_spy * 100, 2),
        "ATR14": round(atr14, 4 if atr14 < 1 else 2),
        "ATR5": round(atr5, 4 if pd.notna(atr5) and atr5 < 1 else 2),
        "ATR20": round(atr20, 4 if pd.notna(atr20) and atr20 < 1 else 2),
        "Gap_%": round(gap_pct, 2) if pd.notna(gap_pct) else np.nan,
        "Score_Top10": score,
        "Score_Top3": round(final_top3_score, 4),
        "Entry_VWAP": round(entry_vwap_pullback, 4 if entry_vwap_pullback < 1 else 2),
        "Entry_Retest": round(entry_breakout_retest, 4 if entry_breakout_retest < 1 else 2),
        "Preferred_Entry": round(preferred_entry, 4 if preferred_entry < 1 else 2),
        "Stop_Price": round(stop_price, 4 if stop_price < 1 else 2),
        "Stop_Limit_Price": round(stop_limit_price, 4 if stop_limit_price < 1 else 2),
        "TP1": round(tp1, 4 if tp1 < 1 else 2),
        "TP2": round(tp2, 4 if tp2 < 1 else 2),
        "TP15": round(tp15, 4 if tp15 < 1 else 2),
        "Notes": ", ".join(notes[:6]),
        "Above_VWAP": above_vwap,
        "SMA200": round(sma200, 4 if sma200 < 1 else 2),
    }


def build_candidate_tables(scan_mode: str, max_records: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Dönüş:
      top10_df: Scanner sonrası en güçlü 10 aday
      top3_df: Bu 10 içinden girişe en uygun 3 aday
      failed_df: İndirilemeyen/elenen loglar
    """
    tv_df = fetch_tv_candidates(scan_mode, max_records=max_records)
    if tv_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ETF ve benzeri ürünleri kaba şekilde temizle.
    tv_df = tv_df[~tv_df.apply(lambda r: likely_non_stock(r["Description"], r["Symbol"]), axis=1)].copy()

    symbols = tv_df["Symbol"].tolist()
    # SPY market context için gerekli.
    symbols_for_download = list(dict.fromkeys(symbols + ["SPY"]))
    history_map, failed_logs = download_yf_history(symbols_for_download, period="260d")

    spy_df = history_map.get("SPY", pd.DataFrame())
    market_ctx = compute_market_context(spy_df)

    evaluated_rows = []
    for _, row in tv_df.iterrows():
        symbol = row["Symbol"]
        hist = history_map.get(symbol)
        if hist is None:
            failed_logs.append({"Hisse": symbol, "Neden": "Geçmiş veri bulunamadı"})
            continue
        result = evaluate_symbol(symbol, row["Description"], hist, market_ctx, scan_mode)
        if result is not None:
            evaluated_rows.append(result)

    all_df = pd.DataFrame(evaluated_rows)
    if all_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(failed_logs)

    # Top 10 aday: scanner gücü yüksek olanlar
    top10_df = all_df.sort_values(["Score_Top10", "Score_Top3"], ascending=[False, False]).head(10).reset_index(drop=True)

    # Top 3 seçimi: fake breakout eleme + final ranking
    eligible = top10_df.copy()
    eligible = eligible[
        (eligible["RVOL"] >= 1.5)
        & (eligible["Close_Strength"] >= 0.60)
        & (eligible["Close"] > eligible["VWAP"])
        & (eligible["Dist_to_20D_High_%"] <= 3.0)
    ]
    eligible = eligible[
        (eligible["Close_Strength"] >= 0.70)
        & (eligible["RS_10D_vs_SPY_%"] > 0)
    ]
    top3_df = eligible.sort_values("Score_Top3", ascending=False).head(3).reset_index(drop=True)

    failed_df = pd.DataFrame(failed_logs)
    return top10_df, top3_df, failed_df


# ============================================================
# SEKME OLUŞTURMA
# ============================================================
tab1, tab2 = st.tabs(["⚡ Canlı Gün İçi Radar", "🔮 Next Day Scanner + En İyi 3"])


# ============================================================
# SEKME 1: CANLI GÜN İÇİ RADAR
# ============================================================
with tab1:
    session_choice = st.radio(
        "Piyasa Oturumu:",
        ["☀️ Gün İçi (Intraday)", "🌅 Piyasa Öncesi (Pre-Market)", "🌙 Kapanış Sonrası (After-Hours)"],
        horizontal=True,
    )

    col_btn, col_chk = st.columns([1, 3])
    with col_btn:
        st.button("🔄 Manuel Yenile", key="btn_refresh_intraday")
    with col_chk:
        auto_refresh = st.checkbox("⚡ 15 saniyede bir otomatik yenile", value=False)

    if auto_refresh:
        st_autorefresh(interval=15000, key="auto_refresh_intraday_v1")

    df_gainers = get_intraday_gainers(session_choice)
    if not df_gainers.empty:
        st.dataframe(df_gainers, use_container_width=True)
    else:
        st.info("Veri bekleniyor veya uygun sonuç bulunamadı.")


# ============================================================
# SEKME 2: NEXT DAY SCANNER + EN İYİ 3
# ============================================================
with tab2:
    st.write("Önce tüm piyasa adayları taranır, sonra Top 10 oluşturulur, ardından içinden girişe en uygun Top 3 seçilir.")
    st.warning("⚠️ Sonuçlar garanti değildir. Önce paper trade ile test et.")

    scan_mode = st.selectbox(
        "Algoritma:",
        [
            "A) Breakout (RVOL≥1.5, güçlü kapanış, VWAP üstü, pozitif RS, kırılıma yakın)",
            "B) Continuation (gap-up, RVOL>2, VWAP üstü kapanış)",
            "C) Accumulation (200MA üstü, hacimli alış, sıkışma, pozitif OBV)",
        ],
    )
    max_scan_records = st.slider("TradingView ilk tarama genişliği", min_value=100, max_value=2000, value=800, step=100)

    if st.button("🚀 Next Day Scanner'ı Başlat"):
        try:
            with st.spinner("Tarama çalışıyor..."):
                top10_df, top3_df, failed_df = build_candidate_tables(scan_mode, max_records=max_scan_records)
                gc.collect()

            if top10_df.empty:
                st.warning("Uygun Top 10 aday bulunamadı.")
            else:
                st.success(f"Top 10 aday bulundu: {len(top10_df)}")
                st.subheader("🏆 Top 10 Aday")
                st.dataframe(top10_df, use_container_width=True)

                if not top3_df.empty:
                    st.subheader("🎯 En İyi 3 (Girişe En Uygun)")
                    st.dataframe(top3_df, use_container_width=True)

                    top3_export = top3_df[[
                        "Symbol", "Category", "Close", "Preferred_Entry", "Stop_Price", "Stop_Limit_Price", "TP1", "TP2", "TP15", "RVOL", "Close_Strength", "RS_10D_vs_SPY_%", "Notes"
                    ]].copy()
                    st.info("Top 3 içinden manuel teyitle işlem yapman daha sağlıklıdır. Öncelik: VWAP pullback veya breakout retest.")
                else:
                    st.warning("Top 10 içinden Top 3 filtresini geçen hisse çıkmadı.")

                csv_top10 = top10_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 Top 10 CSV indir",
                    data=csv_top10,
                    file_name=f"top10_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
                if not top3_df.empty:
                    csv_top3 = top3_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 Top 3 CSV indir",
                        data=csv_top3,
                        file_name=f"top3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

                with st.expander("Reddedilenler / İndirilemeyenler"):
                    if not failed_df.empty:
                        st.dataframe(failed_df, use_container_width=True)
                    else:
                        st.write("Kayıt yok.")
        except Exception as exc:
            st.error(f"Tarama hatası: {exc}")
            if DEBUG_MODE:
                st.code(traceback.format_exc())


st.write("---")


# ============================================================
# VWAP ANALİZİ + ALPACA EMİR MERKEZİ
# ============================================================
st.subheader("📊 VWAP Analizi ve Emir Merkezi")

ticker_input = st.text_input("İşlem yapılacak hisse sembolü", "").upper().strip()

if ticker_input:
    if st.button(f"🔍 {ticker_input} için VWAP analizi yap"):
        with st.spinner("5 dakikalık veriler inceleniyor..."):
            try:
                yf_symbol = normalize_for_yf(ticker_input)
                df_vwap = yf.Ticker(yf_symbol).history(period="1d", interval="5m", prepost=True, auto_adjust=False)
                if not df_vwap.empty:
                    vwap_price = calc_session_vwap(df_vwap)
                    current_price = float(df_vwap["Close"].iloc[-1])
                    day_high = float(df_vwap["High"].max())
                    day_low = float(df_vwap["Low"].min())

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Güncel Fiyat", f"${format_price(current_price)}")
                    c2.metric("Session VWAP", f"${format_price(vwap_price)}")
                    c3.metric("Gün İçi Zirve", f"${format_price(day_high)}")
                    c4.metric("Gün İçi Dip", f"${format_price(day_low)}")

                    st.success(
                        f"Öneri: Fiyat VWAP bölgesine (**${format_price(vwap_price)}**) yakın geri çekilirse limit emir düşün. "
                        "Alternatif olarak önceki kırılım seviyesinin retestini bekle."
                    )
                else:
                    st.warning("İntraday veri bulunamadı.")
            except Exception as exc:
                st.error(f"VWAP analizi hatası: {exc}")
                if DEBUG_MODE:
                    st.code(traceback.format_exc())

api = None
if api_key and secret_key:
    try:
        api = get_api(api_key, secret_key)
    except Exception as exc:
        st.error(f"Alpaca bağlantı hatası: {exc}")

if api is not None:
    with st.form("alpaca_order_form"):
        c1, c2 = st.columns(2)
        with c1:
            qty = st.number_input("Adet", min_value=1, value=100)
            limit_price = st.number_input("Alış limit fiyatı ($)", min_value=0.01, value=1.00, step=0.01)
            ext_hours = st.checkbox("🌙 After-hours çalışsın", value=False)
        with c2:
            take_profit_price = st.number_input("Kar-al fiyatı ($)", min_value=0.01, value=round(limit_price * 1.15, 2), step=0.01)
            stop_loss_price = st.number_input("Zarar-kes fiyatı ($)", min_value=0.01, value=round(limit_price * 0.95, 2), step=0.01)
        submit = st.form_submit_button("🚀 Emri Gönder")

        if submit:
            if not ticker_input:
                st.warning("Önce hisse sembolü gir.")
            elif stop_loss_price >= limit_price:
                st.warning("Stop loss alış limit fiyatından düşük olmalı.")
            elif take_profit_price <= limit_price:
                st.warning("Kar-al fiyatı alış limit fiyatından büyük olmalı.")
            else:
                try:
                    if ext_hours:
                        api.submit_order(
                            symbol=ticker_input,
                            qty=qty,
                            side="buy",
                            type="limit",
                            time_in_force="day",
                            limit_price=round(limit_price, 4),
                            extended_hours=True,
                        )
                        st.success(f"✅ {ticker_input} için after-hours limit emri gönderildi.")
                    else:
                        stop_limit_price = dynamic_stop_limit(stop_loss_price)
                        api.submit_order(
                            symbol=ticker_input,
                            qty=qty,
                            side="buy",
                            type="limit",
                            time_in_force="day",
                            limit_price=round(limit_price, 4),
                            extended_hours=False,
                            order_class="bracket",
                            take_profit={"limit_price": round(take_profit_price, 4)},
                            stop_loss={
                                "stop_price": round(stop_loss_price, 4),
                                "limit_price": round(stop_limit_price, 4),
                            },
                        )
                        st.success(
                            f"✅ {ticker_input} için bracket emir gönderildi. "
                            f"TP=${format_price(take_profit_price)} | SL=${format_price(stop_loss_price)} | StopLimit=${format_price(stop_limit_price)}"
                        )
                except Exception as exc:
                    st.error(f"❌ Alpaca emir hatası: {exc}")
                    if DEBUG_MODE:
                        st.code(traceback.format_exc())
else:
    st.info("Alpaca emir ekranını kullanmak için API anahtarlarını gir.")

st.write("---")
st.caption("Bu araç yatırım tavsiyesi değildir. Önce paper trade ile test etmen gerekir.")
