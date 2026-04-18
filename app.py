import os
import gc
import time
import math
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh


# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(page_title="NextDay Scanner Pro", layout="wide")
st.title("🎯 NextDay Scanner Pro (Kurumsal Motor + Çoklu Seans VWAP)")

DEBUG_MODE = st.sidebar.checkbox("Debug Mode", value=False)

env_api_key = os.getenv("ALPACA_API_KEY", "")
env_secret_key = os.getenv("ALPACA_SECRET_KEY", "")

st.sidebar.header("Alpaca API (Paper)")
api_key = st.sidebar.text_input("API Key ID", value=env_api_key, type="password")
secret_key = st.sidebar.text_input("Secret Key", value=env_secret_key, type="password")

# Ödül filtreleri (ana omurga bozulmadan düşük marjlı hisseleri elemek için)
MIN_TP1_PCT = 5.0
MIN_TP2_PCT = 10.0
TOP3_MIN_TP1_PCT = 6.5
TOP3_MIN_TP2_PCT = 12.0


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================
def log_debug(msg: str):
    if DEBUG_MODE:
        st.sidebar.write(msg)


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        return int(float(x))
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


def get_api(api_key_value, secret_key_value):
    try:
        import alpaca_trade_api as tradeapi
    except ImportError as e:
        raise RuntimeError(
            "alpaca-trade-api paketi kurulu değil. requirements.txt dosyasına alpaca-trade-api ekleyip yeniden deploy et."
        ) from e

    return tradeapi.REST(
        key_id=api_key_value,
        secret_key=secret_key_value,
        base_url="https://paper-api.alpaca.markets",
        api_version="v2",
    )


def get_alpaca_data_api(api_key_value: str, secret_key_value: str):
    if not api_key_value or not secret_key_value:
        return None
    try:
        import alpaca_trade_api as tradeapi
        return tradeapi.REST(
            key_id=api_key_value,
            secret_key=secret_key_value,
            base_url="https://data.alpaca.markets",
            api_version="v2",
        )
    except Exception as exc:
        log_debug(f"Alpaca data api init error: {exc}")
        return None


def get_alpaca_daily_history(ticker: str, api_key_value: str, secret_key_value: str, days_back: int = 420):
    api = get_alpaca_data_api(api_key_value, secret_key_value)
    if api is None:
        return pd.DataFrame()

    try:
        from alpaca_trade_api.rest import TimeFrame
        end_ts = pd.Timestamp.utcnow()
        start_ts = end_ts - pd.Timedelta(days=days_back)
        feed = os.getenv("ALPACA_FEED", "iex") or "iex"
        bars = api.get_bars(
            ticker,
            TimeFrame.Day,
            start=start_ts.isoformat(),
            end=end_ts.isoformat(),
            adjustment="raw",
            feed=feed,
        ).df
        if bars is None or bars.empty:
            return pd.DataFrame()

        if isinstance(bars.index, pd.MultiIndex):
            try:
                bars = bars.xs(ticker, level=0)
            except Exception:
                try:
                    bars = bars.xs(ticker, level="symbol")
                except Exception:
                    pass

        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        bars = bars.rename(columns=rename_map)
        keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in bars.columns]
        bars = bars[keep_cols].dropna()
        return bars
    except Exception as exc:
        log_debug(f"Alpaca daily fetch error for {ticker}: {exc}")
        return pd.DataFrame()

def normalize_symbol_for_yahoo(symbol: str) -> str:
    if not symbol:
        return symbol
    return symbol.replace(".", "-")


def is_likely_non_common_stock(symbol: str, description: str = "") -> bool:
    text = f"{symbol} {description}".upper()

    bad_keywords = [
        " ETF", "ETF ", " FUND", " TRUST", " ETN", "NOTE", "INDEX",
        " ULTRA", " INVERSE", " 2X", " 3X", " LEVERAGED",
        " PREFERRED", " PREF", " ADR", " SPAC", " WARRANT", " RIGHTS"
    ]

    return any(k in text for k in bad_keywords)


def calc_true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = calc_true_range(df)
    return tr.rolling(period).mean()


def calc_closing_strength(last_close: float, last_low: float, last_high: float) -> float:
    daily_range = last_high - last_low
    if daily_range <= 0:
        return np.nan
    return (last_close - last_low) / daily_range


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


def classify_distance_to_entry(current_price: float, reference_entry: float) -> tuple[str, float]:
    if reference_entry <= 0 or not np.isfinite(reference_entry):
        return "NO_TRADE", np.nan

    dist_pct = ((current_price - reference_entry) / reference_entry) * 100
    if dist_pct <= 2.5:
        return "READY", dist_pct
    if dist_pct <= 6.0:
        return "WAIT_RETEST", dist_pct
    return "NO_TRADE", dist_pct


def assess_reward_profile(entry: float, tp1: float, tp2: float) -> dict:
    if entry <= 0 or not np.isfinite(entry):
        return {
            "tp1_pct": np.nan,
            "tp2_pct": np.nan,
            "reward_filter": "INVALID",
            "top3_reward_ok": False,
        }

    tp1_pct = ((tp1 - entry) / entry) * 100 if np.isfinite(tp1) else np.nan
    tp2_pct = ((tp2 - entry) / entry) * 100 if np.isfinite(tp2) else np.nan

    reward_filter = "PASS"
    if pd.isna(tp1_pct) or pd.isna(tp2_pct):
        reward_filter = "INVALID"
    elif tp1_pct < MIN_TP1_PCT or tp2_pct < MIN_TP2_PCT:
        reward_filter = "LOW_REWARD"

    top3_reward_ok = (
        pd.notna(tp1_pct)
        and pd.notna(tp2_pct)
        and tp1_pct >= TOP3_MIN_TP1_PCT
        and tp2_pct >= TOP3_MIN_TP2_PCT
    )

    return {
        "tp1_pct": round(tp1_pct, 2) if pd.notna(tp1_pct) else np.nan,
        "tp2_pct": round(tp2_pct, 2) if pd.notna(tp2_pct) else np.nan,
        "reward_filter": reward_filter,
        "top3_reward_ok": top3_reward_ok,
    }


def compute_single_exit_plan(
    last_close: float,
    last_low: float,
    regular_vwap: float,
    prior_20d_high: float,
    atr14: float,
    breakout_dist: float,
    close_above_vwap: bool,
    closing_strength: float,
    rvol20: float,
    gap_pct: float,
    category: str,
) -> dict:
    atr14 = float(atr14) if pd.notna(atr14) else max(last_close * 0.04, 0.02)
    regular_vwap = float(regular_vwap) if pd.notna(regular_vwap) else last_close
    prior_20d_high = float(prior_20d_high) if pd.notna(prior_20d_high) else last_close

    breakout_entry = max(prior_20d_high * 1.001, regular_vwap * 1.002)
    pullback_entry = max(regular_vwap * 1.001, last_close - 0.35 * atr14)
    reclaim_entry = max(regular_vwap * 1.001, prior_20d_high * 0.998)

    if close_above_vwap and closing_strength >= 0.75 and rvol20 >= 1.5 and breakout_dist <= 0.015:
        entry_type = "BREAKOUT_ENTRY"
        reference_entry = breakout_entry
    elif close_above_vwap and breakout_dist <= 0.06:
        entry_type = "PULLBACK_ENTRY"
        reference_entry = pullback_entry
    elif close_above_vwap and closing_strength >= 0.70:
        entry_type = "RECLAIM_ENTRY"
        reference_entry = reclaim_entry
    else:
        entry_type = "NO_TRADE"
        reference_entry = breakout_entry

    trade_status, dist_pct = classify_distance_to_entry(last_close, reference_entry)

    if entry_type == "NO_TRADE":
        trade_status = "NO_TRADE"

    if gap_pct >= 18 and last_close > reference_entry * 1.04:
        trade_status = "NO_TRADE"
    if last_close > reference_entry * 1.08:
        trade_status = "NO_TRADE"

    base_stop_1 = reference_entry - 1.15 * atr14
    base_stop_2 = regular_vwap * 0.985
    base_stop_3 = last_low * 0.997
    stop_price = max(base_stop_1, base_stop_2, base_stop_3, reference_entry * 0.90)
    stop_price = round(stop_price, 4)
    stop_limit_price = dynamic_stop_limit(stop_price)

    risk_per_share = max(reference_entry - stop_price, reference_entry * 0.02)
    setup_k = 1.6 if category in ["Breakout", "Continuation"] else 1.4
    single_tp = min(reference_entry * 1.12, reference_entry + setup_k * risk_per_share)
    single_tp = round(single_tp, 4)
    rr = round((single_tp - reference_entry) / max(reference_entry - stop_price, 1e-9), 2)

    if trade_status == "NO_TRADE":
        note = "Uzamış/kovalanmamalı"
    elif trade_status == "WAIT_RETEST":
        note = "Geri çekilme bekle"
    else:
        note = "Tek girişe uygun"

    return {
        "entry_type": entry_type,
        "trade_status": trade_status,
        "reference_entry": round(reference_entry, 4),
        "distance_to_entry_pct": round(dist_pct, 2) if pd.notna(dist_pct) else np.nan,
        "stop_price": stop_price,
        "stop_limit_price": stop_limit_price,
        "single_tp": single_tp,
        "risk_reward": rr,
        "plan_note": note,
    }


def calc_position_size(account_size: float, risk_per_trade_pct: float, entry: float, stop: float) -> dict:
    if entry <= 0 or stop <= 0 or entry <= stop:
        return {"shares": 0, "dollar_size": 0, "risk_dollars": 0}

    max_risk_dollars = account_size * risk_per_trade_pct
    risk_per_share = entry - stop
    shares = math.floor(max_risk_dollars / risk_per_share) if risk_per_share > 0 else 0
    dollar_size = shares * entry
    risk_dollars = shares * risk_per_share

    return {
        "shares": max(shares, 0),
        "dollar_size": round(dollar_size, 2),
        "risk_dollars": round(risk_dollars, 2),
    }


# ============================================================
# ÇOKLU SEANS VWAP FONKSİYONLARI
# ============================================================
def split_sessions(intraday_df: pd.DataFrame):
    """
    5 dakikalık veriyi ET saatine göre üçe ayırır:
    - premarket: 04:00 - 09:30
    - regular:   09:30 - 16:00
    - afterhours:16:00 - 20:00
    """
    if intraday_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = intraday_df.copy()

    idx = df.index
    try:
        et_times = idx.tz_convert("America/New_York")
    except Exception:
        try:
            et_times = idx.tz_localize("America/New_York")
        except Exception:
            et_times = idx

    df["et_dt"] = et_times
    df["et_time"] = et_times.time

    premarket = df[
        (df["et_time"] >= pd.to_datetime("04:00").time()) &
        (df["et_time"] < pd.to_datetime("09:30").time())
    ].copy()

    regular = df[
        (df["et_time"] >= pd.to_datetime("09:30").time()) &
        (df["et_time"] < pd.to_datetime("16:00").time())
    ].copy()

    afterhours = df[
        (df["et_time"] >= pd.to_datetime("16:00").time()) &
        (df["et_time"] <= pd.to_datetime("20:00").time())
    ].copy()

    return premarket, regular, afterhours


def calc_session_vwap(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan

    temp = df.dropna(subset=["High", "Low", "Close", "Volume"]).copy()
    if temp.empty:
        return np.nan

    temp["Typical_Price"] = (temp["High"] + temp["Low"] + temp["Close"]) / 3
    temp["VP"] = temp["Typical_Price"] * temp["Volume"]

    vol_sum = temp["Volume"].sum()
    if vol_sum == 0:
        return np.nan

    return float(temp["VP"].sum() / vol_sum)


def session_summary(df: pd.DataFrame):
    if df.empty:
        return {
            "price": np.nan,
            "vwap": np.nan,
            "high": np.nan,
            "low": np.nan,
            "volume": 0,
        }

    return {
        "price": float(df["Close"].iloc[-1]),
        "vwap": calc_session_vwap(df),
        "high": float(df["High"].max()),
        "low": float(df["Low"].min()),
        "volume": int(df["Volume"].sum()),
    }


def get_active_session_et():
    now_et = datetime.now(ZoneInfo("America/New_York")).time()

    if pd.to_datetime("04:00").time() <= now_et < pd.to_datetime("09:30").time():
        return "premarket"
    elif pd.to_datetime("09:30").time() <= now_et < pd.to_datetime("16:00").time():
        return "regular"
    elif pd.to_datetime("16:00").time() <= now_et <= pd.to_datetime("20:00").time():
        return "afterhours"
    else:
        return "closed"


def vwap_decision_engine(price, vwap, high, low, atr14=None, breakout_level=None):
    """
    Aktif seansa göre karar:
    - AL (VWAP DESTEK)
    - BEKLE
    - UZAK DUR
    """
    if pd.isna(price) or pd.isna(vwap) or pd.isna(high) or pd.isna(low):
        return {
            "signal": "NÖTR",
            "entry": None,
            "stop": None,
            "tp1": None,
            "tp2": None,
            "comment": "Veri eksik."
        }

    if vwap <= 0:
        return {
            "signal": "NÖTR",
            "entry": None,
            "stop": None,
            "tp1": None,
            "tp2": None,
            "comment": "VWAP hesaplanamadı."
        }

    distance = (price - vwap) / vwap

    if high - low > 0:
        close_strength = (price - low) / (high - low)
    else:
        close_strength = 0.5

    if pd.isna(atr14) or atr14 <= 0:
        atr14 = max((high - low) * 0.5, price * 0.02)

    if price > vwap and abs(distance) <= 0.01 and close_strength >= 0.6:
        entry = round(vwap * 1.002, 4)
        stop = round(max(vwap - 1.1 * atr14, vwap * 0.98), 4)
        risk = max(entry - stop, 0.01)
        tp1 = round(entry + risk, 4)
        tp2 = round(entry + 2 * risk, 4)

        return {
            "signal": "AL (VWAP DESTEK)",
            "entry": entry,
            "stop": stop,
            "tp1": tp1,
            "tp2": tp2,
            "comment": "Fiyat ilgili seansın VWAP üstünde ve VWAP'a yakın. Kontrollü giriş düşünülebilir."
        }

    if price > vwap and distance > 0.03:
        entry = round(breakout_level, 4) if breakout_level and breakout_level > 0 else round(vwap * 1.01, 4)
        stop = round(max(entry - 1.2 * atr14, entry * 0.97), 4)
        risk = max(entry - stop, 0.01)
        tp1 = round(entry + risk, 4)
        tp2 = round(entry + 2 * risk, 4)

        return {
            "signal": "BEKLE",
            "entry": entry,
            "stop": stop,
            "tp1": tp1,
            "tp2": tp2,
            "comment": "Fiyat ilgili seans VWAP'ından fazla uzaklaşmış. Geri çekilme veya retest beklemek daha doğru."
        }

    if price < vwap:
        return {
            "signal": "UZAK DUR",
            "entry": None,
            "stop": None,
            "tp1": None,
            "tp2": None,
            "comment": "Fiyat ilgili seans VWAP altında. Yapı zayıf."
        }

    return {
        "signal": "NÖTR",
        "entry": None,
        "stop": None,
        "tp1": None,
        "tp2": None,
        "comment": "Net sinyal yok. İzlemeye devam."
    }


# ============================================================
# TRADINGVIEW SCANNER
# ============================================================
TRADINGVIEW_URL = "https://scanner.tradingview.com/america/scan"
TV_HEADERS = {"User-Agent": "Mozilla/5.0"}


def tradingview_scan(
    base_filters: list,
    columns: list,
    sort_field: str,
    max_records: int = 500,
    page_size: int = 100,
) -> list:
    collected = []
    start = 0

    while start < max_records:
        payload = {
            "filter": base_filters,
            "options": {"lang": "en"},
            "markets": ["america"],
            "symbols": {"query": {"types": ["stock"]}, "tickers": []},
            "columns": columns,
            "sort": {"sortBy": sort_field, "sortOrder": "desc"},
            "range": [start, start + page_size - 1],
        }

        try:
            res = requests.post(TRADINGVIEW_URL, json=payload, headers=TV_HEADERS, timeout=20)
            res.raise_for_status()
            data = res.json().get("data", [])
        except Exception as exc:
            log_debug(f"TradingView scan error [{start}-{start+page_size}]: {exc}")
            break

        if not data:
            break

        collected.extend(data)

        if len(data) < page_size:
            break

        start += page_size

    return collected


# ============================================================
# CANLI GÜN İÇİ RADAR
# ============================================================
@st.cache_data(ttl=10)
def get_intraday_gainers(session: str) -> pd.DataFrame:
    try:
        if "Pre-Market" in session:
            sort_field = "premarket_change"
            vol_field = "premarket_volume"
            price_field = "premarket_close"
            min_vol = 10000
        elif "After-Hours" in session:
            sort_field = "postmarket_change"
            vol_field = "postmarket_volume"
            price_field = "postmarket_close"
            min_vol = 10000
        else:
            sort_field = "change"
            vol_field = "volume"
            price_field = "close"
            min_vol = 50000

        payload_filters = [
            {"left": vol_field, "operation": "greater", "right": min_vol},
            {"left": price_field, "operation": "greater", "right": 0.50},
            {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
        ]

        items = tradingview_scan(
            base_filters=payload_filters,
            columns=["name", "description", price_field, sort_field, vol_field],
            sort_field=sort_field,
            max_records=25,
            page_size=25,
        )

        results = []
        for item in items[:15]:
            try:
                d = item["d"]
                symbol = d[0]
                company = d[1] or ""

                if is_likely_non_common_stock(symbol, company):
                    continue

                price = safe_float(d[2], 0)
                price = round(price, 4) if price < 1 else round(price, 2)

                results.append(
                    {
                        "Hisse": symbol,
                        "Şirket": company,
                        "Fiyat ($)": price,
                        "Artış (%)": round(safe_float(d[3], 0), 2),
                        "Hacim": f"{safe_int(d[4], 0):,}",
                    }
                )
            except Exception as exc:
                log_debug(f"Intraday parse error: {exc}")

        return pd.DataFrame(results)

    except Exception as exc:
        log_debug(f"get_intraday_gainers error: {exc}")
        return pd.DataFrame()


# ============================================================
# SWING RADAR - AŞAMA 1
# ============================================================
@st.cache_data(ttl=600)
def fetch_tradingview_candidates(algo_choice: str, max_records: int = 500) -> pd.DataFrame:
    base_filters = [
        {"left": "close", "operation": "greater", "right": 2.00},
        {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
        {"left": "volume", "operation": "greater", "right": 250000},
    ]

    if "A)" in algo_choice:
        algo_filters = [
            {"left": "relative_volume_10d_calc", "operation": "greater", "right": 1.5},
        ]
        sort_field = "relative_volume_10d_calc"
    elif "B)" in algo_choice:
        algo_filters = [
            {"left": "gap", "operation": "greater", "right": 2.0},
            {"left": "relative_volume_10d_calc", "operation": "greater", "right": 2.0},
        ]
        sort_field = "gap"
    else:
        algo_filters = []
        sort_field = "volume"

    columns = [
        "name",
        "description",
        "close",
        "volume",
        "relative_volume_10d_calc",
        "gap",
        "market_cap_basic",
    ]

    raw = tradingview_scan(
        base_filters=base_filters + algo_filters,
        columns=columns,
        sort_field=sort_field,
        max_records=max_records,
        page_size=100,
    )

    rows = []
    seen = set()

    for item in raw:
        try:
            d = item["d"]
            symbol = d[0]
            description = d[1] or ""

            if not symbol or symbol in seen:
                continue

            if is_likely_non_common_stock(symbol, description):
                continue

            seen.add(symbol)
            rows.append(
                {
                    "symbol": symbol,
                    "yahoo_symbol": normalize_symbol_for_yahoo(symbol),
                    "description": description,
                    "tv_close": safe_float(d[2], np.nan),
                    "tv_volume": safe_float(d[3], np.nan) if len(d) > 3 else np.nan,
                    "tv_rvol": safe_float(d[4], np.nan) if len(d) > 4 else np.nan,
                    "tv_gap": safe_float(d[5], np.nan) if len(d) > 5 else np.nan,
                    "tv_market_cap": safe_float(d[6], np.nan) if len(d) > 6 else np.nan,
                }
            )
        except Exception as exc:
            log_debug(f"Candidate parse error: {exc}")

    return pd.DataFrame(rows)


# ============================================================
# YFINANCE VERİ İNDİRME (Anti-Bot Hayalet Modu)
# ============================================================
@st.cache_data(ttl=120)
def download_daily_data_chunked(tickers: list[str], period: str = "220d", chunk_size: int = 10, pause: float = 2.0):
    if not tickers:
        return {}

    data_dict = {}

    def _normalize_sub(df_like):
        if df_like is None or df_like.empty:
            return pd.DataFrame()
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df_like.columns]
        if len(cols) < 5:
            return pd.DataFrame()
        out = df_like[cols].dropna()
        return out if not out.empty else pd.DataFrame()

    def _single_fetch(ticker: str):
        # 1) yfinance.download
        try:
            sub = yf.download(
                tickers=ticker,
                period=period,
                progress=False,
                threads=False,
                auto_adjust=False,
                group_by="ticker",
            )
            sub = _normalize_sub(sub)
            if not sub.empty:
                return sub
        except Exception as exc:
            log_debug(f"Single yf.download error for {ticker}: {exc}")

        # 2) yfinance.Ticker.history
        try:
            hist = yf.Ticker(ticker).history(
                period=period,
                interval="1d",
                auto_adjust=False,
                prepost=False,
            )
            hist = _normalize_sub(hist)
            if not hist.empty:
                return hist
        except Exception as exc:
            log_debug(f"Ticker.history error for {ticker}: {exc}")

        return pd.DataFrame()

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            yf_data = yf.download(
                tickers=chunk,
                period=period,
                progress=False,
                threads=False,
                auto_adjust=False,
                group_by="ticker",
            )

            if isinstance(yf_data.columns, pd.MultiIndex):
                for ticker in chunk:
                    try:
                        sub = pd.DataFrame({
                            "Open": yf_data[ticker]["Open"],
                            "High": yf_data[ticker]["High"],
                            "Low": yf_data[ticker]["Low"],
                            "Close": yf_data[ticker]["Close"],
                            "Volume": yf_data[ticker]["Volume"],
                        }).dropna()
                        if not sub.empty:
                            data_dict[ticker] = sub
                    except Exception:
                        pass
            elif len(chunk) == 1:
                sub = _normalize_sub(yf_data)
                if not sub.empty:
                    data_dict[chunk[0]] = sub

        except Exception as exc:
            log_debug(f"Chunk download error: {chunk[:3]}... -> {exc}")

        missing = [ticker for ticker in chunk if ticker not in data_dict]
        for ticker in missing:
            sub = _single_fetch(ticker)
            if not sub.empty:
                data_dict[ticker] = sub
            time.sleep(0.8)

        time.sleep(pause)

    return data_dict

@st.cache_data(ttl=300)
def get_intraday_session_data(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        intraday = stock.history(period="1d", interval="5m", prepost=True, auto_adjust=False)
        if intraday.empty:
            return pd.DataFrame()
        return intraday
    except Exception as exc:
        log_debug(f"Intraday data error for {ticker}: {exc}")
        return pd.DataFrame()


# ============================================================
# SWING RADAR - AŞAMA 2
# ============================================================
def evaluate_candidates(algo_choice: str, tv_candidates_df: pd.DataFrame, data_dict: dict, api_key_value: str = "", secret_key_value: str = ""):
    final_candidates = []
    rejected_log = []

    if tv_candidates_df.empty:
        return final_candidates, rejected_log

    spy_df = data_dict.get("SPY")
    if spy_df is None or spy_df.empty:
        spy_df = get_alpaca_daily_history("SPY", api_key_value, secret_key_value, days_back=420)
        if spy_df is not None and not spy_df.empty:
            data_dict["SPY"] = spy_df

    spy_ret_10d = 0.0
    if spy_df is not None and len(spy_df) >= 11:
        spy_ret_10d = (spy_df["Close"].iloc[-1] - spy_df["Close"].iloc[-10]) / spy_df["Close"].iloc[-10]

    for _, row in tv_candidates_df.iterrows():
        symbol = row["symbol"]
        yahoo_symbol = row["yahoo_symbol"]
        description = row.get("description", "")

        try:
            df = data_dict.get(yahoo_symbol)

            if df is None or df.empty:
                df = get_alpaca_daily_history(yahoo_symbol, api_key_value, secret_key_value, days_back=420)
                if df is not None and not df.empty:
                    data_dict[yahoo_symbol] = df
                    rejected_note = None
                else:
                    rejected_log.append({"Hisse": symbol, "Neden": "Yahoo veri yok / Alpaca fallback başarısız"})
                    continue

            if len(df) < 220:
                rejected_log.append({"Hisse": symbol, "Neden": "Yetersiz günlük veri (<220)"})
                continue

            last_close = float(df["Close"].iloc[-1])
            last_open = float(df["Open"].iloc[-1])
            last_high = float(df["High"].iloc[-1])
            last_low = float(df["Low"].iloc[-1])
            last_volume = float(df["Volume"].iloc[-1])

            df = df.copy()
            df["ATR14"] = calc_atr(df, 14)
            df["ATR5"] = calc_atr(df, 5)
            df["ATR20"] = calc_atr(df, 20)

            atr14 = float(df["ATR14"].iloc[-1]) if not pd.isna(df["ATR14"].iloc[-1]) else np.nan
            atr5 = float(df["ATR5"].iloc[-1]) if not pd.isna(df["ATR5"].iloc[-1]) else np.nan
            atr20 = float(df["ATR20"].iloc[-1]) if not pd.isna(df["ATR20"].iloc[-1]) else np.nan

            closing_strength = calc_closing_strength(last_close, last_low, last_high)

            stock_ret_10d = (df["Close"].iloc[-1] - df["Close"].iloc[-10]) / df["Close"].iloc[-10]
            rs_positive = stock_ret_10d > spy_ret_10d
            rs_spread = stock_ret_10d - spy_ret_10d

            prior_20d_high = df["High"].shift(1).rolling(20).max().iloc[-1]
            breakout_dist = (
                (prior_20d_high - last_close) / last_close
                if pd.notna(prior_20d_high) and prior_20d_high > last_close
                else 0
            )

            sma50 = df["Close"].rolling(50).mean().iloc[-1]
            sma200 = df["Close"].rolling(200).mean().iloc[-1]

            avg_vol_20 = df["Volume"].rolling(20).mean().iloc[-1]
            rvol20 = last_volume / avg_vol_20 if avg_vol_20 and avg_vol_20 > 0 else np.nan

            prev_close = df["Close"].iloc[-2]
            gap_pct = ((last_open - prev_close) / prev_close) * 100 if prev_close > 0 else 0

            up_days = df[df["Close"] > df["Open"]].tail(15)
            down_days = df[df["Close"] < df["Open"]].tail(15)
            up_vol = up_days["Volume"].mean() if not up_days.empty else 0
            down_vol = down_days["Volume"].mean() if not down_days.empty else 0

            close_diff = df["Close"].diff().fillna(0)
            obv = np.where(close_diff > 0, df["Volume"], np.where(close_diff < 0, -df["Volume"], 0))
            df["OBV"] = pd.Series(obv, index=df.index).cumsum()
            obv_slope_10 = df["OBV"].iloc[-1] - df["OBV"].iloc[-10]

            intraday = get_intraday_session_data(yahoo_symbol)
            _, regular_df, _ = split_sessions(intraday)
            regular_vwap = calc_session_vwap(regular_df) if not regular_df.empty else np.nan
            if pd.isna(regular_vwap):
                regular_vwap = (last_high + last_low + last_close) / 3

            close_above_vwap = last_close > regular_vwap

            score = 0
            notes = []

            if pd.notna(rvol20):
                if rvol20 >= 5:
                    score += 30
                    notes.append("RVOL 5x+")
                elif rvol20 >= 3:
                    score += 24
                    notes.append("RVOL 3x+")
                elif rvol20 >= 2:
                    score += 18
                    notes.append("RVOL 2x+")
                elif rvol20 >= 1.5:
                    score += 10
                    notes.append("RVOL 1.5x+")

            if pd.notna(closing_strength):
                if closing_strength >= 0.9:
                    score += 20
                    notes.append("Tepe kapanış")
                elif closing_strength >= 0.75:
                    score += 14
                    notes.append("Güçlü kapanış")
                elif closing_strength >= 0.6:
                    score += 8

            if close_above_vwap:
                score += 10
                notes.append("VWAP üstü")
            else:
                score -= 6

            if rs_positive:
                score += 10
                notes.append("SPY'ye göre güçlü")
            else:
                score -= 5

            if pd.notna(sma200) and last_close > sma200:
                score += 8
                notes.append("200MA üstü")

            if pd.notna(atr5) and pd.notna(atr20):
                if atr5 < atr20 * 0.85:
                    score += 8
                    notes.append("Sıkışma")
                elif atr5 > atr20 * 1.15:
                    score += 4
                    notes.append("ATR genişliyor")

            if breakout_dist <= 0.02:
                score += 10
                notes.append("Kırılıma <%2")
            elif breakout_dist <= 0.05:
                score += 4

            if obv_slope_10 > 0:
                score += 6
                notes.append("OBV pozitif")

            category = None

            if "A)" in algo_choice:
                if (
                    pd.notna(rvol20)
                    and rvol20 >= 1.5
                    and pd.notna(closing_strength)
                    and closing_strength >= 0.75
                    and close_above_vwap
                    and rs_positive
                    and breakout_dist <= 0.02
                ):
                    category = "Breakout"

            elif "B)" in algo_choice:
                if (
                    gap_pct >= 2.0
                    and pd.notna(rvol20)
                    and rvol20 >= 2.0
                    and close_above_vwap
                    and pd.notna(closing_strength)
                    and closing_strength >= 0.70
                ):
                    category = "Continuation"

            elif "C)" in algo_choice:
                if (
                    pd.notna(sma200)
                    and last_close > sma200
                    and up_vol > down_vol * 1.2
                    and pd.notna(atr5)
                    and pd.notna(atr20)
                    and atr5 < atr20
                    and obv_slope_10 > 0
                ):
                    category = "Accumulation"

            if category:
                trade_plan = compute_single_exit_plan(
                    last_close=last_close,
                    last_low=last_low,
                    regular_vwap=regular_vwap,
                    prior_20d_high=prior_20d_high,
                    atr14=atr14,
                    breakout_dist=breakout_dist,
                    close_above_vwap=close_above_vwap,
                    closing_strength=closing_strength,
                    rvol20=rvol20,
                    gap_pct=gap_pct,
                    category=category,
                )
                tp1 = round(trade_plan["reference_entry"] + (trade_plan["reference_entry"] - trade_plan["stop_price"]), 4)
                tp2 = round(trade_plan["reference_entry"] + 2 * (trade_plan["reference_entry"] - trade_plan["stop_price"]), 4)
                reward_profile = assess_reward_profile(trade_plan["reference_entry"], tp1, tp2)

                final_trade_status = trade_plan["trade_status"]
                plan_note = trade_plan["plan_note"]
                if reward_profile["reward_filter"] == "LOW_REWARD":
                    final_trade_status = "NO_TRADE_LOW_REWARD"
                    plan_note = "Getiri marjı düşük; zorlama aday değil"

                final_candidates.append(
                    {
                        "Symbol": symbol,
                        "Yahoo_Symbol": yahoo_symbol,
                        "Description": description,
                        "Category": category,
                        "Move_Type": row.get("move_type", "MIXED"),
                        "Entry_Type": trade_plan["entry_type"],
                        "Trade_Status": final_trade_status,
                        "Close": round(last_close, 4 if last_close < 1 else 2),
                        "RVOL": round(rvol20, 2) if pd.notna(rvol20) else np.nan,
                        "Close_Strength": round(closing_strength, 2) if pd.notna(closing_strength) else np.nan,
                        "Dist_to_High_%": round(breakout_dist * 100, 2),
                        "VWAP_Regular": round(regular_vwap, 4 if regular_vwap < 1 else 2),
                        "Above_VWAP": bool(close_above_vwap),
                        "RS_10d_minus_SPY_%": round(rs_spread * 100, 2),
                        "ATR14": round(atr14, 4 if pd.notna(atr14) and atr14 < 1 else 2) if pd.notna(atr14) else np.nan,
                        "Gap_%": round(gap_pct, 2),
                        "SMA50": round(sma50, 2) if pd.notna(sma50) else np.nan,
                        "SMA200": round(sma200, 2) if pd.notna(sma200) else np.nan,
                        "OBV_Positive": bool(obv_slope_10 > 0),
                        "Entry_Idea": trade_plan["reference_entry"],
                        "Distance_to_Entry_%": trade_plan["distance_to_entry_pct"],
                        "Stop_Price": trade_plan["stop_price"],
                        "Stop_Limit_Price": trade_plan["stop_limit_price"],
                        "Single_TP": trade_plan["single_tp"],
                        "Risk_Reward": trade_plan["risk_reward"],
                        "TP1": tp1,
                        "TP2": tp2,
                        "TP1_%": reward_profile["tp1_pct"],
                        "TP2_%": reward_profile["tp2_pct"],
                        "Reward_Filter": reward_profile["reward_filter"],
                        "Top3_Reward_OK": reward_profile["top3_reward_ok"],
                        "Score": score,
                        "Notes": ", ".join(notes[:5] + [plan_note]),
                    }
                )
            else:
                rejected_log.append({"Hisse": symbol, "Neden": "İkinci aşama kuralları geçilemedi"})

        except Exception as exc:
            rejected_log.append({"Hisse": symbol, "Neden": f"Hata: {str(exc)}"})
            log_debug(f"Evaluate error {symbol}: {exc}")

    final_candidates = sorted(final_candidates, key=lambda x: x["Score"], reverse=True)
    return final_candidates, rejected_log


# ============================================================
# TOP 10 -> EN İYİ 3
# ============================================================
def rank_top3(candidates_df: pd.DataFrame) -> pd.DataFrame:
    if candidates_df.empty:
        return pd.DataFrame()

    df = candidates_df.copy()

    df["rvol_score"] = np.minimum(df["RVOL"].fillna(0) / 3.0, 1.0)
    df["close_strength_score"] = df["Close_Strength"].fillna(0).clip(0, 1)
    df["vwap_score"] = np.where(df["Above_VWAP"] == True, 1.0, 0.0)
    df["breakout_score"] = 1 - np.minimum(df["Dist_to_High_%"].fillna(999) / 3.0, 1.0)
    df["compression_score"] = np.where(df["Dist_to_High_%"].fillna(99) <= 2.0, 0.8, 0.5)
    df["rs_score"] = np.clip(df["RS_10d_minus_SPY_%"].fillna(0) / 10.0, 0, 1)
    df["clean_chart"] = np.where(df["Close_Strength"].fillna(0) >= 0.7, 1.0, 0.5)
    df["entry_quality_score"] = np.select(
        [df["Trade_Status"] == "READY", df["Trade_Status"] == "WAIT_RETEST"],
        [1.0, 0.65],
        default=0.0,
    )
    df["rr_score"] = np.clip(df["Risk_Reward"].fillna(0) / 2.0, 0, 1)
    df["reward_score"] = np.where(df["Top3_Reward_OK"] == True, 1.0, 0.0)

    df["final_score"] = (
        0.22 * df["rvol_score"] +
        0.18 * df["close_strength_score"] +
        0.12 * df["breakout_score"] +
        0.10 * df["compression_score"] +
        0.10 * df["vwap_score"] +
        0.10 * df["rs_score"] +
        0.08 * df["clean_chart"] +
        0.10 * df["entry_quality_score"] +
        0.10 * df["reward_score"]
    )

    df = df[
        (df["RVOL"].fillna(0) >= 1.5) &
        (df["Close_Strength"].fillna(0) >= 0.6) &
        (df["Above_VWAP"] == True) &
        (df["Dist_to_High_%"].fillna(999) <= 3.5) &
        (~df["Trade_Status"].isin(["NO_TRADE", "NO_TRADE_LOW_REWARD"])) &
        (df["Reward_Filter"] == "PASS")
    ]

    df = df[
        (df["Close_Strength"].fillna(0) >= 0.7) &
        (df["RS_10d_minus_SPY_%"].fillna(-999) > 0) &
        (df["Top3_Reward_OK"] == True)
    ]

    df["stability_bonus"] = np.where(df["Close"] >= 5, 0.05, 0.0)
    df["final_score"] = df["final_score"] + df["stability_bonus"] + (0.05 * df["rr_score"])

    df = df.sort_values(["Trade_Status", "final_score", "Score"], ascending=[True, False, False])
    status_order = pd.Categorical(df["Trade_Status"], categories=["READY", "WAIT_RETEST", "NO_TRADE_LOW_REWARD", "NO_TRADE"], ordered=True)
    df = df.assign(_status_order=status_order).sort_values(["_status_order", "final_score", "Score"], ascending=[True, False, False]).drop(columns=["_status_order"])
    return df.head(3)


# ============================================================
# EKRANLAR
# ============================================================
tab1, tab2, tab3 = st.tabs(
    [
        "⚡ Canlı Gün İçi Radar",
        "🔮 Kurumsal Swing Radar",
        "📊 VWAP Analizi ve Emir Merkezi",
    ]
)


# ============================================================
# TAB 1 - CANLI GÜN İÇİ RADAR
# ============================================================
with tab1:
    session_choice = st.radio(
        "Piyasa Oturumu:",
        ["☀️ Gün İçi (Intraday)", "🌅 Piyasa Öncesi (Pre-Market)", "🌙 Kapanış Sonrası (After-Hours)"],
        horizontal=True,
    )

    col_btn, col_chk = st.columns([1, 3])

    with col_btn:
        st.button("🔄 Manuel Yenile", key="btn_refresh_1")

    with col_chk:
        auto_refresh = st.checkbox("⚡ 15 saniyede bir otomatik yenile", value=False)

    if auto_refresh:
        st_autorefresh(interval=15000, key="auto_refresh_gainers")

    df_gainers = get_intraday_gainers(session_choice)

    if not df_gainers.empty:
        st.dataframe(df_gainers, use_container_width=True)
    else:
        st.info("Veri bekleniyor veya uygun hisse bulunamadı.")


# ============================================================
# TAB 2 - SWING RADAR
# ============================================================
if "swing_scan_results" not in st.session_state:
    st.session_state.swing_scan_results = None

if "swing_scan_error" not in st.session_state:
    st.session_state.swing_scan_error = None

with tab2:
    st.write("Profesyonel filtrelere göre ertesi gün potansiyeli yüksek adaylar:")

    algo_choice = st.selectbox(
        "Algoritma:",
        [
            "A) Keskin Nişancı Breakout (RVOL≥1.5, Kapanış>%75, VWAP Üstü, Pozitif RS, Kırılıma <%2)",
            "B) İkinci Gün Koşusu (Gap-Up, RVOL>2, VWAP Üstü Kapanış)",
            "C) Kurumsal Birikim (200MA Üstü, Hacimli Alışlar, Sıkışma, Pozitif OBV)",
        ],
    )

    max_scan_records = st.slider(
        "İlk tarama genişliği",
        min_value=100,
        max_value=1200,
        value=500,
        step=100,
    )

    if st.button("🚀 Tarayıcıyı Başlat"):
        st.session_state.swing_scan_error = None
        st.session_state.swing_scan_results = None

        try:
            with st.spinner("1. Aşama: TradingView adayları taranıyor..."):
                tv_df = fetch_tradingview_candidates(algo_choice=algo_choice, max_records=max_scan_records)

            result_payload = {
                "tv_count": 0,
                "final_df": pd.DataFrame(),
                "top3_df": pd.DataFrame(),
                "rejected_log": pd.DataFrame(),
                "message_type": "info",
                "message": "",
            }

            if tv_df.empty:
                result_payload["message_type"] = "warning"
                result_payload["message"] = "İlk aşamada aday bulunamadı."
            else:
                yahoo_tickers = tv_df["yahoo_symbol"].dropna().unique().tolist()
                if "SPY" not in yahoo_tickers:
                    yahoo_tickers.append("SPY")

                with st.spinner("2. Aşama: Günlük veriler chunk'lı indiriliyor..."):
                    data_dict = download_daily_data_chunked(
                        yahoo_tickers,
                        period="220d",
                        chunk_size=10,
                        pause=2.0,
                    )

                with st.spinner("3. Aşama: İkinci filtre ve scoring uygulanıyor..."):
                    final_candidates, rejected_log = evaluate_candidates(algo_choice, tv_df, data_dict, api_key, secret_key)

                gc.collect()

                final_df = pd.DataFrame(final_candidates)
                top3_df = rank_top3(final_df.head(10)) if not final_df.empty else pd.DataFrame()
                rej_df = pd.DataFrame(rejected_log)

                result_payload["tv_count"] = len(tv_df)
                result_payload["final_df"] = final_df
                result_payload["top3_df"] = top3_df
                result_payload["rejected_log"] = rej_df

                yahoo_fail_count = 0
                if not rej_df.empty and "Neden" in rej_df.columns:
                    yahoo_fail_count = rej_df["Neden"].astype(str).str.contains("Yahoo veri yok", na=False).sum()
                yahoo_fail_ratio = (yahoo_fail_count / max(len(tv_df), 1)) * 100
                result_payload["yahoo_fail_count"] = int(yahoo_fail_count)
                result_payload["yahoo_fail_ratio"] = round(yahoo_fail_ratio, 1)

                if final_df.empty:
                    result_payload["message_type"] = "warning"
                    if yahoo_fail_ratio >= 40:
                        result_payload["message"] = (
                            f"Kurallara uyan aday çıkmadı. Veri katmanında sorun olabilir: "
                            f"{yahoo_fail_count}/{len(tv_df)} adayda Yahoo/Alpaca günlük veri alınamadı."
                        )
                    else:
                        result_payload["message"] = "Kurallara uyan aday çıkmadı."
                else:
                    result_payload["message_type"] = "success"
                    suffix = ""
                    if yahoo_fail_ratio >= 20:
                        suffix = f" | Veri uyarısı: {yahoo_fail_count}/{len(tv_df)} adayda Yahoo/Alpaca veri sorunu"
                    result_payload["message"] = f"Filtrelerden başarıyla geçen hisse sayısı: {len(final_df)}{suffix}"

            st.session_state.swing_scan_results = result_payload

        except Exception as exc:
            st.session_state.swing_scan_error = traceback.format_exc() if DEBUG_MODE else str(exc)

    if st.session_state.swing_scan_error:
        st.error(f"Tarama hatası: {st.session_state.swing_scan_error}")

    if st.session_state.swing_scan_results is not None:
        result_payload = st.session_state.swing_scan_results

        if result_payload["tv_count"] > 0:
            st.info(f"İlk aşamada bulunan aday sayısı: {result_payload['tv_count']}")
            yahoo_fail_ratio = result_payload.get("yahoo_fail_ratio", 0)
            yahoo_fail_count = result_payload.get("yahoo_fail_count", 0)
            if yahoo_fail_ratio >= 40:
                st.error(f"Veri kaynağı uyarısı: {yahoo_fail_count}/{result_payload['tv_count']} adayda Yahoo/Alpaca günlük veri alınamadı.")
            elif yahoo_fail_ratio >= 20:
                st.warning(f"Veri kaynağı uyarısı: {yahoo_fail_count}/{result_payload['tv_count']} adayda Yahoo/Alpaca günlük veri alınamadı.")

        msg_type = result_payload["message_type"]
        msg_text = result_payload["message"]

        if msg_type == "warning":
            st.warning(msg_text)
        elif msg_type == "success":
            st.success(msg_text)
        elif msg_type == "info":
            st.info(msg_text)

        final_df = result_payload["final_df"]
        top3_df = result_payload["top3_df"]
        rej_df = result_payload["rejected_log"]

        if not final_df.empty:
            st.subheader("🏆 Top 10 Aday")
            st.dataframe(final_df.head(10), use_container_width=True)

            st.subheader("🥇 En İyi 3")
            if not top3_df.empty:
                st.dataframe(top3_df, use_container_width=True)
            else:
                st.warning("Top 10 içinden entry-ready 3 hisse çıkmadı.")

            csv_all = final_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 Tüm sonuçları CSV indir",
                data=csv_all,
                file_name=f"nextday_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

            if not top3_df.empty:
                csv_top3 = top3_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 En iyi 3 CSV indir",
                    data=csv_top3,
                    file_name=f"nextday_top3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

        with st.expander("Reddedilenler / Hata kayıtları"):
            if not rej_df.empty:
                st.dataframe(rej_df, use_container_width=True)
            else:
                st.write("Kayıt yok.")


# ============================================================
# TAB 3 - ÇOKLU SEANS VWAP + EMİR MERKEZİ
# ============================================================
with tab3:
    st.subheader("📊 VWAP Analizi ve Emir Merkezi")

    ticker_input = st.text_input("İşlem yapılacak hisse sembolü", "").upper().strip()
    account_size = st.number_input("Hesap büyüklüğü ($)", min_value=100.0, value=2000.0, step=100.0)
    risk_pct = st.number_input("Trade başına risk (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.1) / 100

    if ticker_input:
        if st.button(f"🔍 {ticker_input} için çoklu seans VWAP analizi yap"):
            with st.spinner("Veriler çekiliyor ve seans bazlı VWAP hesaplanıyor..."):
                try:
                    yahoo_symbol = normalize_symbol_for_yahoo(ticker_input)

                    stock = yf.Ticker(yahoo_symbol)
                    intraday = stock.history(period="1d", interval="5m", prepost=True, auto_adjust=False)

                    daily_dict = download_daily_data_chunked([yahoo_symbol], period="260d", chunk_size=1, pause=0)
                    daily_df = daily_dict.get(yahoo_symbol, pd.DataFrame())

                    if intraday.empty or daily_df.empty:
                        st.warning("Analiz için yeterli veri bulunamadı.")
                    else:
                        premarket_df, regular_df, afterhours_df = split_sessions(intraday)

                        premarket_info = session_summary(premarket_df)
                        regular_info = session_summary(regular_df)
                        afterhours_info = session_summary(afterhours_df)

                        daily_df = daily_df.copy()
                        daily_df["ATR14"] = calc_atr(daily_df, 14)
                        atr14 = float(daily_df["ATR14"].iloc[-1]) if not pd.isna(daily_df["ATR14"].iloc[-1]) else np.nan
                        breakout_level = daily_df["High"].shift(1).rolling(20).max().iloc[-1]

                        active_session = get_active_session_et()

                        if active_session == "premarket":
                            active_info = premarket_info
                            session_name = "Pre-Market"
                        elif active_session == "regular":
                            active_info = regular_info
                            session_name = "Regular Session"
                        elif active_session == "afterhours":
                            active_info = afterhours_info
                            session_name = "After-Hours"
                        else:
                            active_info = regular_info
                            session_name = "Market Closed (referans: Regular Session)"

                        result = vwap_decision_engine(
                            price=active_info["price"],
                            vwap=active_info["vwap"],
                            high=active_info["high"],
                            low=active_info["low"],
                            atr14=atr14,
                            breakout_level=breakout_level
                        )

                        st.markdown("### 📌 Çoklu Seans VWAP Paneli")
                        st.info(f"Aktif seans: **{session_name}**")

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Pre-Market VWAP", format_price(premarket_info["vwap"]))
                        c2.metric("Regular VWAP", format_price(regular_info["vwap"]))
                        c3.metric("After-Hours VWAP", format_price(afterhours_info["vwap"]))

                        c4, c5, c6 = st.columns(3)
                        c4.metric("Pre-Market Hacim", f"{premarket_info['volume']:,}")
                        c5.metric("Regular Hacim", f"{regular_info['volume']:,}")
                        c6.metric("After-Hours Hacim", f"{afterhours_info['volume']:,}")

                        c7, c8, c9 = st.columns(3)
                        c7.metric("Pre-Market Son Fiyat", format_price(premarket_info["price"]))
                        c8.metric("Regular Son Fiyat", format_price(regular_info["price"]))
                        c9.metric("After-Hours Son Fiyat", format_price(afterhours_info["price"]))

                        st.markdown("### 🎯 Aktif Seans Kararı")

                        if result["signal"].startswith("AL"):
                            st.success(result["signal"])
                        elif result["signal"] == "BEKLE":
                            st.warning(result["signal"])
                        elif result["signal"] == "UZAK DUR":
                            st.error(result["signal"])
                        else:
                            st.info(result["signal"])

                        st.write(result["comment"])

                        if result["entry"] is not None:
                            pos = calc_position_size(
                                account_size=account_size,
                                risk_per_trade_pct=risk_pct,
                                entry=result["entry"],
                                stop=result["stop"],
                            )

                            plan_df = pd.DataFrame([{
                                "Ticker": ticker_input,
                                "Aktif Seans": session_name,
                                "Signal": result["signal"],
                                "Entry": result["entry"],
                                "Stop": result["stop"],
                                "TP1": result["tp1"],
                                "TP2": result["tp2"],
                                "Önerilen Adet": pos["shares"],
                                "Pozisyon Büyüklüğü ($)": pos["dollar_size"],
                                "Risk ($)": pos["risk_dollars"],
                            }])

                            st.dataframe(plan_df, use_container_width=True)

                            st.info(
                                f"Öneri: {ticker_input} için yaklaşık {pos['shares']} adet, "
                                f"${format_price(result['entry'])} giriş, "
                                f"${format_price(result['stop'])} stop."
                            )

                except Exception as exc:
                    st.error(f"VWAP analizi sırasında hata oluştu: {exc}")
                    if DEBUG_MODE:
                        st.code(traceback.format_exc())

    st.write("---")
    st.markdown("### 🚀 Emir Gönder")

    if api_key and secret_key:
        try:
            api = get_api(api_key, secret_key)
        except Exception as exc:
            st.error(f"Alpaca bağlantı hatası: {exc}")
            api = None
    else:
        api = None

    if api is None:
        st.info("Emir ekranını kullanmak için Alpaca API key bilgilerini gir.")
    else:
        with st.form("bracket_order_form"):
            col1, col2 = st.columns(2)

            with col1:
                qty = st.number_input("Adet", min_value=1, value=100)
                limit_price = st.number_input("Alış limit fiyatı ($)", min_value=0.01, value=1.00, step=0.01)
                ext_hours = st.checkbox("🌙 After-hours çalışsın", value=False)

            with col2:
                take_profit_price = st.number_input("Kar-al fiyatı ($)", min_value=0.01, value=1.15, step=0.01)
                stop_loss_price = st.number_input("Zarar-kes fiyatı ($)", min_value=0.01, value=0.95, step=0.01)

            submit_button = st.form_submit_button("🚀 Emri Gönder")

            if submit_button:
                if not ticker_input:
                    st.warning("Önce hisse sembolü gir.")
                elif stop_loss_price >= limit_price:
                    st.warning("Stop loss alış limit fiyatından küçük olmalı.")
                elif take_profit_price <= limit_price:
                    st.warning("Take profit alış limit fiyatından büyük olmalı.")
                else:
                    try:
                        order_symbol = normalize_symbol_for_yahoo(ticker_input)

                        if ext_hours:
                            api.submit_order(
                                symbol=order_symbol,
                                qty=int(qty),
                                side="buy",
                                type="limit",
                                time_in_force="day",
                                limit_price=round(limit_price, 4),
                                extended_hours=True,
                            )
                            st.success(
                                f"✅ {order_symbol} için ${format_price(limit_price)} seviyesinden "
                                f"after-hours limit emir iletildi."
                            )
                        else:
                            stop_limit_price = dynamic_stop_limit(stop_loss_price)

                            api.submit_order(
                                symbol=order_symbol,
                                qty=int(qty),
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
                                f"✅ {order_symbol} için bracket emir iletildi. "
                                f"Giriş=${format_price(limit_price)} | "
                                f"TP=${format_price(take_profit_price)} | "
                                f"SL=${format_price(stop_loss_price)} | "
                                f"StopLimit=${format_price(stop_limit_price)}"
                            )

                    except Exception as exc:
                        st.error(f"❌ Alpaca emir hatası: {exc}")
                        if DEBUG_MODE:
                            st.code(traceback.format_exc())


st.write("---")
st.caption(
    "Bu araç yatırım tavsiyesi değildir. Önce paper trading ile test edilmesi gerekir. "
    "Dış veri kaynaklarında rate-limit ve veri uyuşmazlığı olabilir."
)
