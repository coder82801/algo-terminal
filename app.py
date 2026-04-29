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

st.sidebar.caption("Günlük veri kaynağı: Önce Alpaca historical bars, eksik kalırsa Yahoo fallback.")

st.sidebar.divider()
st.sidebar.header("Son Filtre")
REJECT_EXTENDED = st.sidebar.checkbox("EXTENDED adayları ele", value=True)
MIN_FINAL_CONFIDENCE = st.sidebar.slider("Min nihai confidence", min_value=0, max_value=100, value=60, step=5)

# Core guardrails
MAX_EXTENSION_ABOVE_BREAKOUT_PCT = 0.08   # %8 üstü: breakout değil, extended kabul et
MAX_CONTINUATION_EXTENSION_PCT = 0.06     # continuation için daha sıkı üst sınır


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
    except ImportError as exc:
        raise RuntimeError("alpaca-trade-api paketi kurulu değil.") from exc

    return tradeapi.REST(
        key_id=api_key_value,
        secret_key=secret_key_value,
        base_url="https://paper-api.alpaca.markets",
        api_version="v2",
    )


def _normalize_yf_period(period: str) -> str:
    supported = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
    if period in supported:
        return period
    if isinstance(period, str) and period.endswith("d"):
        try:
            days = int(period[:-1])
            if days >= 200:
                return "1y"
            if days >= 120:
                return "6mo"
            if days >= 60:
                return "3mo"
        except Exception:
            pass
    return "1y"


def _alpaca_headers(api_key_value: str, secret_key_value: str) -> dict:
    return {
        "APCA-API-KEY-ID": api_key_value,
        "APCA-API-SECRET-KEY": secret_key_value,
    }


def fetch_alpaca_daily_single(symbol: str, api_key_value: str, secret_key_value: str, feed: str = "iex") -> pd.DataFrame:
    if not api_key_value or not secret_key_value:
        return pd.DataFrame()

    start = (datetime.now(ZoneInfo("UTC")) - pd.Timedelta(days=430)).strftime("%Y-%m-%dT00:00:00Z")
    end = (datetime.now(ZoneInfo("UTC")) + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Day",
        "start": start,
        "end": end,
        "adjustment": "all",
        "limit": 1000,
        "feed": feed or "iex",
    }

    try:
        res = requests.get(url, headers=_alpaca_headers(api_key_value, secret_key_value), params=params, timeout=20)
        if res.status_code != 200:
            log_debug(f"Alpaca daily {symbol} -> HTTP {res.status_code}: {res.text[:200]}")
            return pd.DataFrame()

        payload = res.json()
        bars = payload.get("bars", [])
        if not bars:
            return pd.DataFrame()

        rows = []
        for bar in bars:
            rows.append({
                "Date": pd.to_datetime(bar.get("t"), utc=True),
                "Open": safe_float(bar.get("o")),
                "High": safe_float(bar.get("h")),
                "Low": safe_float(bar.get("l")),
                "Close": safe_float(bar.get("c")),
                "Volume": safe_float(bar.get("v")),
            })

        df = pd.DataFrame(rows).dropna()
        if df.empty:
            return df

        df = df.set_index("Date").sort_index()
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as exc:
        log_debug(f"Alpaca daily fetch error for {symbol}: {exc}")
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
# YENİ: ENTRY TYPE + MOVE TYPE ENGINE
# ============================================================
def close_enough(value: float, threshold: float) -> bool:
    try:
        return value <= threshold
    except Exception:
        return False


def classify_entry_type(
    close_above_vwap: bool,
    dist_below_breakout: float,
    extension_above_breakout: float,
    rvol20: float,
    gap_pct: float,
    closing_strength: float,
    atr5: float,
    atr20: float,
):
    if not close_above_vwap:
        return "WEAK"

    if extension_above_breakout > MAX_EXTENSION_ABOVE_BREAKOUT_PCT:
        return "EXTENDED"

    atr_expanding = pd.notna(atr5) and pd.notna(atr20) and atr5 > atr20 * 1.10

    if dist_below_breakout <= 0.02 and extension_above_breakout <= 0.03 and rvol20 >= 2.0 and closing_strength >= 0.70:
        return "BREAKOUT"

    if 0.02 < dist_below_breakout <= 0.05 and extension_above_breakout <= 0.02 and rvol20 >= 1.5 and closing_strength >= 0.60:
        return "MICRO_PULLBACK"

    if close_above_vwap and extension_above_breakout <= 0.02 and gap_pct >= 0 and closing_strength >= 0.60:
        return "VWAP_PULLBACK"

    if close_above_vwap and dist_below_breakout <= 0.01 and extension_above_breakout <= 0.03 and rvol20 >= 3 and atr_expanding:
        return "BREAKOUT"

    return "EXTENDED"


def classify_move_type(
    gap_pct: float,
    rvol20: float,
    closing_strength: float,
    obv_slope_10: float,
    dist_below_breakout: float,
    extension_above_breakout: float,
    atr5: float,
    atr20: float,
    last_close: float,
    sma50: float,
):
    atr_expanding = pd.notna(atr5) and pd.notna(atr20) and atr5 > atr20 * 1.15
    trend_ok = pd.notna(sma50) and last_close > sma50

    if extension_above_breakout > 0.10:
        return "WEAK_MOVE"

    if gap_pct >= 3.0 and rvol20 >= 2.0 and closing_strength >= 0.75 and obv_slope_10 > 0:
        return "NEWS_DRIVEN"

    if rvol20 >= 4.0 and atr_expanding and closing_strength >= 0.70 and dist_below_breakout <= 0.03 and extension_above_breakout <= 0.05:
        return "SHORT_SQUEEZE"

    if close_enough(dist_below_breakout, 0.05) and extension_above_breakout <= 0.05 and rvol20 >= 1.5 and trend_ok and obv_slope_10 > 0:
        return "TECHNICAL_MOMENTUM"

    return "WEAK_MOVE"


def compute_confidence_score(
    rvol20: float,
    closing_strength: float,
    close_above_vwap: bool,
    rs_positive: bool,
    dist_below_breakout: float,
    extension_above_breakout: float,
    move_type: str,
    entry_type: str,
):
    score = 0

    if pd.notna(rvol20):
        if rvol20 >= 5:
            score += 25
        elif rvol20 >= 3:
            score += 20
        elif rvol20 >= 2:
            score += 14
        elif rvol20 >= 1.5:
            score += 8

    if pd.notna(closing_strength):
        if closing_strength >= 0.9:
            score += 20
        elif closing_strength >= 0.75:
            score += 15
        elif closing_strength >= 0.60:
            score += 8

    if close_above_vwap:
        score += 10

    if rs_positive:
        score += 10

    if dist_below_breakout <= 0.02 and extension_above_breakout <= 0.03:
        score += 12
    elif dist_below_breakout <= 0.05 and extension_above_breakout <= 0.05:
        score += 6

    if extension_above_breakout > 0.08:
        score -= 12
    elif extension_above_breakout > 0.05:
        score -= 6

    if move_type == "NEWS_DRIVEN":
        score += 12
    elif move_type == "SHORT_SQUEEZE":
        score += 10
    elif move_type == "TECHNICAL_MOMENTUM":
        score += 8

    if entry_type == "BREAKOUT":
        score += 10
    elif entry_type == "MICRO_PULLBACK":
        score += 7
    elif entry_type == "VWAP_PULLBACK":
        score += 5
    elif entry_type == "EXTENDED":
        score -= 8
    elif entry_type == "WEAK":
        score -= 15

    return max(0, min(100, int(round(score))))


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
# YFINANCE VERİ İNDİRME
# ============================================================
@st.cache_data(ttl=300)
def download_daily_data_chunked(
    tickers: list[str],
    period: str = "220d",
    chunk_size: int = 10,
    pause: float = 2.0,
    alpaca_key: str = "",
    alpaca_secret: str = "",
    alpaca_feed: str = "iex",
):
    if not tickers:
        return {}

    data_dict = {}
    yf_period = _normalize_yf_period(period)

    def _single_yf_fetch(ticker: str):
        try:
            sub = yf.download(
                tickers=ticker,
                period=yf_period,
                progress=False,
                threads=False,
                auto_adjust=False,
                group_by="ticker",
            )
            if sub is not None and not sub.empty:
                sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if not sub.empty:
                    return sub
        except Exception as exc:
            log_debug(f"Single Yahoo fetch error for {ticker}: {exc}")
        return pd.DataFrame()

    if alpaca_key and alpaca_secret:
        for ticker in tickers:
            sub = fetch_alpaca_daily_single(ticker, alpaca_key, alpaca_secret, alpaca_feed)
            if not sub.empty:
                data_dict[ticker] = sub
            time.sleep(0.15)

    remaining = [ticker for ticker in tickers if ticker not in data_dict]
    for i in range(0, len(remaining), chunk_size):
        chunk = remaining[i:i + chunk_size]
        try:
            yf_data = yf.download(
                tickers=chunk,
                period=yf_period,
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
                        continue
            elif len(chunk) == 1 and yf_data is not None and not yf_data.empty:
                sub = yf_data[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if not sub.empty:
                    data_dict[chunk[0]] = sub
        except Exception as exc:
            log_debug(f"Yahoo chunk download error {chunk[:3]}... -> {exc}")

        missing = [ticker for ticker in chunk if ticker not in data_dict]
        for ticker in missing:
            sub = _single_yf_fetch(ticker)
            if not sub.empty:
                data_dict[ticker] = sub
            time.sleep(0.5)

        time.sleep(pause)

    return data_dict


def fetch_alpaca_intraday_5m(symbol: str, api_key_value: str, secret_key_value: str, feed: str = "iex") -> pd.DataFrame:
    if not api_key_value or not secret_key_value:
        return pd.DataFrame()

    end = datetime.now(ZoneInfo("UTC"))
    start = end - pd.Timedelta(days=7)

    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "5Min",
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "adjustment": "all",
        "limit": 10000,
        "feed": feed or "iex",
    }

    try:
        res = requests.get(
            url,
            headers=_alpaca_headers(api_key_value, secret_key_value),
            params=params,
            timeout=20,
        )
        if res.status_code != 200:
            log_debug(f"Alpaca intraday {symbol} -> HTTP {res.status_code}: {res.text[:200]}")
            return pd.DataFrame()

        payload = res.json()
        bars = payload.get("bars", [])
        if not bars:
            return pd.DataFrame()

        rows = []
        for bar in bars:
            rows.append({
                "Date": pd.to_datetime(bar.get("t"), utc=True),
                "Open": safe_float(bar.get("o")),
                "High": safe_float(bar.get("h")),
                "Low": safe_float(bar.get("l")),
                "Close": safe_float(bar.get("c")),
                "Volume": safe_float(bar.get("v")),
            })

        df = pd.DataFrame(rows).dropna()
        if df.empty:
            return df

        df = df.set_index("Date").sort_index()
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as exc:
        log_debug(f"Alpaca intraday fetch error for {symbol}: {exc}")
        return pd.DataFrame()


def _dynamic_min_regular_bars(now_et: datetime | None = None) -> int:
    """Seansın ne kadar ilerlediğine göre minimum gerekli regular 5m bar sayısı.
    Günün erken saatlerinde 50 bar beklemek intraday modülü gereksiz yere kilitler.
    """
    now_et = now_et or datetime.now(ZoneInfo("America/New_York"))
    session_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    minutes_since_open = max(0, int((now_et - session_open).total_seconds() // 60))

    if minutes_since_open < 20:
        return 3
    if minutes_since_open < 45:
        return 6
    if minutes_since_open < 90:
        return 9
    if minutes_since_open < 150:
        return 12
    if minutes_since_open < 240:
        return 18
    return 24



def pick_last_valid_regular_day(intraday_df: pd.DataFrame, min_regular_bars: int | None = None) -> pd.DataFrame:
    """Regular session barları gerçekten olan en son geçerli günü seçer.
    Varsayılan eşik seans saatine göre dinamik belirlenir.
    """
    if intraday_df is None or intraday_df.empty:
        return pd.DataFrame()

    if min_regular_bars is None:
        min_regular_bars = _dynamic_min_regular_bars()

    df = intraday_df.copy()
    idx = df.index
    try:
        et_index = idx.tz_convert("America/New_York")
    except Exception:
        try:
            et_index = idx.tz_localize("America/New_York")
        except Exception:
            et_index = idx

    unique_days = sorted(pd.Index(et_index.date).unique(), reverse=True)
    if len(unique_days) == 0:
        return pd.DataFrame()

    for day in unique_days:
        mask = pd.Index(et_index.date) == day
        sub = df.loc[mask].copy()
        if sub.empty:
            continue
        _, regular_df, _ = split_sessions(sub)
        reg_vol = regular_df["Volume"].fillna(0).sum() if not regular_df.empty else 0
        if not regular_df.empty and len(regular_df) >= min_regular_bars and reg_vol > 0:
            return sub

    return pd.DataFrame()


def _extract_last_active_day(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    idx = out.index
    try:
        et_index = idx.tz_convert("America/New_York")
    except Exception:
        try:
            et_index = idx.tz_localize("America/New_York")
        except Exception:
            et_index = idx

    out["et_index"] = et_index
    out["et_date"] = pd.Index(et_index.date)

    valid_dates = list(pd.Series(out["et_date"]).dropna().unique())
    if not valid_dates:
        return pd.DataFrame()

    last_day = valid_dates[-1]
    out = out[out["et_date"] == last_day].copy()
    out = out.drop(columns=["et_index", "et_date"], errors="ignore")
    return out


@st.cache_data(ttl=120)
def _fetch_yahoo_intraday_candidates(yahoo_symbol: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    try:
        stock = yf.Ticker(yahoo_symbol)
        df_5m = stock.history(period="5d", interval="5m", prepost=True, auto_adjust=False)
        if df_5m is not None and not df_5m.empty:
            out["5m_5d"] = df_5m[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
    except Exception as exc:
        log_debug(f"Yahoo 5m intraday error for {yahoo_symbol}: {exc}")

    try:
        stock = yf.Ticker(yahoo_symbol)
        df_1m = stock.history(period="2d", interval="1m", prepost=True, auto_adjust=False)
        if df_1m is not None and not df_1m.empty:
            out["1m_2d"] = df_1m[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
    except Exception as exc:
        log_debug(f"Yahoo 1m intraday error for {yahoo_symbol}: {exc}")

    return out


@st.cache_data(ttl=120)
def get_intraday_session_data(
    symbol: str,
    yahoo_symbol: str,
    api_key_value: str = "",
    secret_key_value: str = "",
    alpaca_feed: str = "iex",
):
    min_bars = _dynamic_min_regular_bars()

    intraday = fetch_alpaca_intraday_5m(symbol, api_key_value, secret_key_value, alpaca_feed)
    intraday = pick_last_valid_regular_day(intraday, min_regular_bars=min_bars)
    if not intraday.empty:
        return intraday

    yahoo_candidates = _fetch_yahoo_intraday_candidates(yahoo_symbol)
    for key in ["5m_5d", "1m_2d"]:
        cand = yahoo_candidates.get(key, pd.DataFrame())
        cand = pick_last_valid_regular_day(cand, min_regular_bars=min_bars)
        if not cand.empty:
            return cand

    return pd.DataFrame()


def get_intraday_session_data_verbose(
    symbol: str,
    yahoo_symbol: str,
    api_key_value: str = "",
    secret_key_value: str = "",
    alpaca_feed: str = "iex",
):
    min_bars = _dynamic_min_regular_bars()

    alpaca_raw = fetch_alpaca_intraday_5m(symbol, api_key_value, secret_key_value, alpaca_feed)
    if alpaca_raw is not None and not alpaca_raw.empty:
        picked = pick_last_valid_regular_day(alpaca_raw, min_regular_bars=min_bars)
        if not picked.empty:
            return picked, "alpaca_ok"
        return pd.DataFrame(), f"Alpaca veri var ama geçerli regular session yok (<{min_bars} bar veya hacim yok)"

    yahoo_candidates = _fetch_yahoo_intraday_candidates(yahoo_symbol)
    if not yahoo_candidates:
        return pd.DataFrame(), "Alpaca boş, Yahoo boş"

    for key in ["5m_5d", "1m_2d"]:
        cand = yahoo_candidates.get(key, pd.DataFrame())
        if cand is None or cand.empty:
            continue
        picked = pick_last_valid_regular_day(cand, min_regular_bars=min_bars)
        if not picked.empty:
            return picked, f"yahoo_{key}_ok"

    detail = []
    if "5m_5d" in yahoo_candidates:
        detail.append("Yahoo 5m var ama geçerli regular session yok")
    if "1m_2d" in yahoo_candidates:
        detail.append("Yahoo 1m var ama geçerli regular session yok")
    if detail:
        return pd.DataFrame(), "; ".join(detail) + f" (<{min_bars} bar veya hacim yok)"

    return pd.DataFrame(), "Intraday veri yok"


def _get_spy_return_10d(data_dict: dict, alpaca_key: str, alpaca_secret: str, alpaca_feed: str) -> tuple[float, bool]:
    spy_df = data_dict.get("SPY")
    if spy_df is not None and not spy_df.empty and len(spy_df) >= 11:
        ret = (float(spy_df["Close"].iloc[-1]) - float(spy_df["Close"].iloc[-10])) / float(spy_df["Close"].iloc[-10])
        return ret, True

    log_debug("SPY verisi eksik; tekrar indiriliyor...")

    if alpaca_key and alpaca_secret:
        spy_df = fetch_alpaca_daily_single("SPY", alpaca_key, alpaca_secret, alpaca_feed)
        if not spy_df.empty and len(spy_df) >= 11:
            ret = (float(spy_df["Close"].iloc[-1]) - float(spy_df["Close"].iloc[-10])) / float(spy_df["Close"].iloc[-10])
            return ret, True

    try:
        spy = yf.download("SPY", period="1mo", progress=False, threads=False, auto_adjust=False)
        if spy is not None and not spy.empty:
            close = spy["Close"].dropna()
            if len(close) >= 11:
                ret = (float(close.iloc[-1]) - float(close.iloc[-10])) / float(close.iloc[-10])
                return ret, True
    except Exception as exc:
        log_debug(f"SPY retry failed: {exc}")

    return 0.0, False


# ============================================================
# SWING RADAR - AŞAMA 2
# ============================================================
def evaluate_candidates(algo_choice: str, tv_candidates_df: pd.DataFrame, data_dict: dict):
    final_candidates = []
    rejected_log = []

    if tv_candidates_df.empty:
        return final_candidates, rejected_log

    spy_ret_10d, spy_ok = _get_spy_return_10d(
        data_dict=data_dict,
        alpaca_key=api_key,
        alpaca_secret=secret_key,
        alpaca_feed=os.getenv("ALPACA_FEED", "iex"),
    )

    if not spy_ok:
        log_debug("SPY verisi sağlıklı alınamadı; RS metrikleri temkinli çalışacak.")

    for _, row in tv_candidates_df.iterrows():
        symbol = row["symbol"]
        yahoo_symbol = row["yahoo_symbol"]
        description = row.get("description", "")

        try:
            df = data_dict.get(yahoo_symbol)

            if df is None or df.empty:
                rejected_log.append({"Hisse": symbol, "Neden": "Yahoo/Alpaca günlük veri yok"})
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
            rs_positive = stock_ret_10d > spy_ret_10d if spy_ok else False
            rs_spread = stock_ret_10d - spy_ret_10d if spy_ok else np.nan

            prior_20d_high = df["High"].shift(1).rolling(20).max().iloc[-1]

            dist_below_breakout = 0.0
            extension_above_breakout = 0.0
            if pd.notna(prior_20d_high) and prior_20d_high > 0:
                if last_close < prior_20d_high:
                    dist_below_breakout = (prior_20d_high - last_close) / last_close
                elif last_close > prior_20d_high:
                    extension_above_breakout = (last_close - prior_20d_high) / prior_20d_high

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

            intraday = get_intraday_session_data(
                symbol=symbol,
                yahoo_symbol=yahoo_symbol,
                api_key_value=api_key,
                secret_key_value=secret_key,
                alpaca_feed=os.getenv("ALPACA_FEED", "iex"),
            )

            if intraday.empty:
                rejected_log.append({"Hisse": symbol, "Neden": "Intraday veri yok / geçerli regular session bulunamadı"})
                continue

            _, regular_df, _ = split_sessions(intraday)
            if regular_df.empty or len(regular_df) < 50:
                rejected_log.append({"Hisse": symbol, "Neden": "Regular session barları yetersiz"})
                continue

            regular_vwap = calc_session_vwap(regular_df)

            if pd.isna(regular_vwap) or regular_vwap <= 0:
                rejected_log.append({"Hisse": symbol, "Neden": "Regular VWAP hesaplanamadı"})
                continue

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

            if dist_below_breakout <= 0.02 and extension_above_breakout <= 0.03:
                score += 10
                notes.append("Kırılıma yakın")
            elif dist_below_breakout <= 0.05 and extension_above_breakout <= 0.05:
                score += 4
            elif extension_above_breakout > 0.08:
                score -= 10
                notes.append("Aşırı uzamış")

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
                    and obv_slope_10 > 0
                    and dist_below_breakout <= 0.02
                    and extension_above_breakout <= 0.03
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
                    and rs_positive
                    and obv_slope_10 > 0
                    and extension_above_breakout <= MAX_CONTINUATION_EXTENSION_PCT
                    and (dist_below_breakout <= 0.03 or extension_above_breakout <= 0.05)
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
                entry_type = classify_entry_type(
                    close_above_vwap=close_above_vwap,
                    dist_below_breakout=dist_below_breakout,
                    extension_above_breakout=extension_above_breakout,
                    rvol20=rvol20 if pd.notna(rvol20) else 0,
                    gap_pct=gap_pct,
                    closing_strength=closing_strength if pd.notna(closing_strength) else 0,
                    atr5=atr5,
                    atr20=atr20,
                )

                move_type = classify_move_type(
                    gap_pct=gap_pct,
                    rvol20=rvol20 if pd.notna(rvol20) else 0,
                    closing_strength=closing_strength if pd.notna(closing_strength) else 0,
                    obv_slope_10=obv_slope_10,
                    dist_below_breakout=dist_below_breakout,
                    extension_above_breakout=extension_above_breakout,
                    atr5=atr5,
                    atr20=atr20,
                    last_close=last_close,
                    sma50=sma50,
                )

                confidence = compute_confidence_score(
                    rvol20=rvol20 if pd.notna(rvol20) else 0,
                    closing_strength=closing_strength if pd.notna(closing_strength) else 0,
                    close_above_vwap=close_above_vwap,
                    rs_positive=rs_positive,
                    dist_below_breakout=dist_below_breakout,
                    extension_above_breakout=extension_above_breakout,
                    move_type=move_type,
                    entry_type=entry_type,
                )

                if REJECT_EXTENDED and entry_type == "EXTENDED":
                    rejected_log.append({"Hisse": symbol, "Neden": "Entry_Type=EXTENDED"})
                    continue

                if confidence < MIN_FINAL_CONFIDENCE:
                    rejected_log.append({"Hisse": symbol, "Neden": f"Confidence<{MIN_FINAL_CONFIDENCE}"})
                    continue

                if entry_type == "BREAKOUT":
                    if pd.notna(prior_20d_high):
                        if last_close > prior_20d_high:
                            entry_idea = round(last_close * 1.001, 4)
                        else:
                            entry_idea = round(max(last_close, prior_20d_high * 1.002), 4)
                    else:
                        entry_idea = round(last_close, 4)
                elif entry_type == "MICRO_PULLBACK":
                    entry_idea = round(max(regular_vwap * 1.01, last_close * 0.995), 4)
                elif entry_type == "VWAP_PULLBACK":
                    entry_idea = round(regular_vwap * 1.002, 4) if close_above_vwap else round(last_close, 4)
                elif entry_type == "EXTENDED":
                    entry_idea = round(last_close, 4)
                else:
                    entry_idea = round(last_close, 4)

                if pd.notna(atr14):
                    stop_price = round(max(entry_idea - 1.2 * atr14, entry_idea * 0.95), 4)
                else:
                    stop_price = round(entry_idea * 0.95, 4)

                if stop_price >= entry_idea:
                    stop_price = round(entry_idea * 0.95, 4)

                stop_limit_price = dynamic_stop_limit(stop_price)
                risk = max(entry_idea - stop_price, 0.01)

                if entry_type == "BREAKOUT":
                    tp1 = round(entry_idea + 1.5 * risk, 4)
                    tp2 = round(entry_idea + 3.0 * risk, 4)
                elif entry_type == "MICRO_PULLBACK":
                    tp1 = round(entry_idea + 1.3 * risk, 4)
                    tp2 = round(entry_idea + 2.6 * risk, 4)
                elif entry_type == "VWAP_PULLBACK":
                    tp1 = round(entry_idea + 1.2 * risk, 4)
                    tp2 = round(entry_idea + 2.4 * risk, 4)
                else:
                    tp1 = round(entry_idea + risk, 4)
                    tp2 = round(entry_idea + 2 * risk, 4)

                final_candidates.append(
                    {
                        "Symbol": symbol,
                        "Yahoo_Symbol": yahoo_symbol,
                        "Description": description,
                        "Category": category,
                        "Move_Type": move_type,
                        "Entry_Type": entry_type,
                        "Confidence": confidence,
                        "Close": round(last_close, 4 if last_close < 1 else 2),
                        "RVOL": round(rvol20, 2) if pd.notna(rvol20) else np.nan,
                        "Close_Strength": round(closing_strength, 2) if pd.notna(closing_strength) else np.nan,
                        "Dist_to_High_%": round(dist_below_breakout * 100, 2),
                        "Extension_Above_High_%": round(extension_above_breakout * 100, 2),
                        "VWAP_Regular": round(regular_vwap, 4 if regular_vwap < 1 else 2),
                        "Above_VWAP": bool(close_above_vwap),
                        "RS_10d_minus_SPY_%": round(rs_spread * 100, 2) if pd.notna(rs_spread) else np.nan,
                        "ATR14": round(atr14, 4 if pd.notna(atr14) and atr14 < 1 else 2) if pd.notna(atr14) else np.nan,
                        "Gap_%": round(gap_pct, 2),
                        "SMA50": round(sma50, 2) if pd.notna(sma50) else np.nan,
                        "SMA200": round(sma200, 2) if pd.notna(sma200) else np.nan,
                        "OBV_Positive": bool(obv_slope_10 > 0),
                        "Entry_Idea": entry_idea,
                        "Stop_Price": stop_price,
                        "Stop_Limit_Price": stop_limit_price,
                        "TP1": tp1,
                        "TP2": tp2,
                        "Score": score,
                        "Notes": ", ".join(notes[:6]),
                    }
                )
            else:
                rejected_log.append({"Hisse": symbol, "Neden": "İkinci aşama kuralları geçilemedi"})

        except Exception as exc:
            rejected_log.append({"Hisse": symbol, "Neden": f"Hata: {str(exc)}"})
            log_debug(f"Evaluate error {symbol}: {exc}")

    final_candidates = sorted(
        final_candidates,
        key=lambda x: (x["Confidence"], x["Score"]),
        reverse=True
    )
    return final_candidates, rejected_log


# ============================================================
# TOP 3
# ============================================================
def rank_top3(candidates_df: pd.DataFrame) -> pd.DataFrame:
    if candidates_df.empty:
        return pd.DataFrame()

    df = candidates_df.copy()

    df["rvol_score"] = np.minimum(df["RVOL"].fillna(0) / 3.0, 1.0)
    df["close_strength_score"] = df["Close_Strength"].fillna(0).clip(0, 1)
    df["vwap_score"] = np.where(df["Above_VWAP"] == True, 1.0, 0.0)
    df["breakout_score"] = 1 - np.minimum(df["Dist_to_High_%"].fillna(999) / 3.0, 1.0)
    df["extension_penalty"] = np.where(df["Extension_Above_High_%"].fillna(0) > 5, -0.12, 0.0)
    df["compression_score"] = np.where(df["Dist_to_High_%"].fillna(99) <= 2.0, 0.8, 0.5)
    df["rs_score"] = np.clip(df["RS_10d_minus_SPY_%"].fillna(0) / 10.0, 0, 1)
    df["clean_chart"] = np.where(df["Close_Strength"].fillna(0) >= 0.7, 1.0, 0.5)

    entry_bonus_map = {
        "BREAKOUT": 0.12,
        "MICRO_PULLBACK": 0.08,
        "VWAP_PULLBACK": 0.06,
        "EXTENDED": -0.05,
        "WEAK": -0.12,
    }
    df["entry_bonus"] = df["Entry_Type"].map(entry_bonus_map).fillna(0)

    move_bonus_map = {
        "NEWS_DRIVEN": 0.10,
        "SHORT_SQUEEZE": 0.08,
        "TECHNICAL_MOMENTUM": 0.06,
        "WEAK_MOVE": -0.04,
    }
    df["move_bonus"] = df["Move_Type"].map(move_bonus_map).fillna(0)

    df["confidence_score"] = np.clip(df["Confidence"].fillna(0) / 100.0, 0, 1)

    df["final_score"] = (
        0.20 * df["rvol_score"] +
        0.18 * df["close_strength_score"] +
        0.12 * df["breakout_score"] +
        0.10 * df["compression_score"] +
        0.10 * df["vwap_score"] +
        0.10 * df["rs_score"] +
        0.08 * df["clean_chart"] +
        0.07 * df["confidence_score"] +
        df["entry_bonus"] +
        df["move_bonus"] +
        df["extension_penalty"]
    )

    df = df[
        (df["RVOL"].fillna(0) >= 1.5) &
        (df["Close_Strength"].fillna(0) >= 0.6) &
        (df["Above_VWAP"] == True) &
        (df["RS_10d_minus_SPY_%"].fillna(-999) > 0) &
        (df["Extension_Above_High_%"].fillna(999) <= 8.0)
    ]

    df["stability_bonus"] = np.where(df["Close"] >= 5, 0.05, 0.0)
    df["final_score"] = df["final_score"] + df["stability_bonus"]

    df = df.sort_values(["final_score", "Confidence", "Score"], ascending=False)
    return df.head(3)



# ============================================================
# INTRADAY TRADE ENGINE
# ============================================================
def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(series: pd.Series):
    ema12 = calc_ema(series, 12)
    ema26 = calc_ema(series, 26)
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def calc_intraday_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    temp = df.copy()
    prev_close = temp['Close'].shift(1)
    tr = pd.concat([
        temp['High'] - temp['Low'],
        (temp['High'] - prev_close).abs(),
        (temp['Low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_intraday_setup(features: dict) -> str:
    # Önce acımasız kırmızı bayraklar
    if features.get("vwap_lost_after_open", False):
        return "NO_SETUP"
    if features.get("opening_range_fail", False):
        return "NO_SETUP"
    if features.get("price_to_premarket_high_pct", np.nan) < -4.0:
        return "NO_SETUP"
    if features.get("price_to_day_high_pct", np.nan) < -2.5:
        return "NO_SETUP"
    if features.get("session_rvol", np.nan) < 1.2:
        return "NO_SETUP"

    # 1) Açılış sürüşü: gap + hacim + ORB + VWAP üstü + tepeye yakın kalma
    if (
        features["close_above_vwap"]
        and features["opening_range_break"]
        and features["ema9_above_ema20"]
        and features["macd_hist_positive"]
        and features["gap_pct"] >= 3.0
        and features["session_rvol"] >= 2.0
        and features["price_to_day_high_pct"] >= -1.0
        and features["price_to_premarket_high_pct"] >= -2.0
    ):
        return "OPEN_DRIVE_BREAKOUT"

    # 2) İlk pullback: güçlü hisse, VWAP/EMA9 üstünde kontrollü geri çekilme
    if (
        features["close_above_vwap"]
        and features["ema9_above_ema20"]
        and features["macd_hist_positive"]
        and features["first_pullback_hold"]
        and features["session_rvol"] >= 1.5
        and features["price_to_vwap_pct"] >= -0.2
        and features["price_to_ema9_pct"] >= -0.4
        and features["price_to_day_high_pct"] >= -1.5
        and features["price_to_premarket_high_pct"] >= -3.0
    ):
        return "FIRST_PULLBACK"

    # 3) VWAP reclaim: gün içi zayıflamadan sonra reclaim ama yalnızca tekrar güç varsa
    if (
        features["close_above_vwap"]
        and features["macd_hist_positive"]
        and features["ema9_above_ema20"]
        and features["vwap_reclaim"]
        and features["session_rvol"] >= 1.3
        and features["rsi14"] >= 60
        and features["price_to_day_high_pct"] >= -2.0
        and features["price_to_premarket_high_pct"] >= -3.5
    ):
        return "VWAP_RECLAIM"

    return "NO_SETUP"


def intraday_confidence_score(features: dict, setup_type: str) -> int:
    score = 0

    srvol = features.get("session_rvol", np.nan)
    if pd.notna(srvol):
        if srvol >= 3.0:
            score += 26
        elif srvol >= 2.0:
            score += 20
        elif srvol >= 1.5:
            score += 14
        elif srvol >= 1.2:
            score += 8

    if features["close_above_vwap"]:
        score += 14
    if features["ema9_above_ema20"]:
        score += 10
    if features["ema20_above_ema50"]:
        score += 6
    if features["macd_hist_positive"]:
        score += 8

    rsi = features["rsi14"]
    if pd.notna(rsi):
        if 60 <= rsi <= 75:
            score += 10
        elif 55 <= rsi < 60:
            score += 5
        elif rsi > 82:
            score -= 8
        elif rsi < 50:
            score -= 12

    if features["opening_range_break"]:
        score += 10
    if features["first_pullback_hold"]:
        score += 8
    if features["vwap_reclaim"]:
        score += 6

    gap_pct = features.get("gap_pct", np.nan)
    if pd.notna(gap_pct):
        if gap_pct >= 8:
            score += 10
        elif gap_pct >= 3:
            score += 6
        elif gap_pct < -5:
            score -= 6

    rs_intraday = features.get("rs_vs_spy_intraday_pct", 0.0)
    if rs_intraday >= 5:
        score += 10
    elif rs_intraday >= 2:
        score += 6
    elif rs_intraday < 0:
        score -= 8

    pth = features.get("price_to_day_high_pct", np.nan)
    if pd.notna(pth):
        if pth >= -0.75:
            score += 10
        elif pth >= -1.5:
            score += 6
        elif pth < -3.0:
            score -= 14
        elif pth < -2.0:
            score -= 8

    ptph = features.get("price_to_premarket_high_pct", np.nan)
    if pd.notna(ptph):
        if ptph >= -1.5:
            score += 8
        elif ptph < -4.0:
            score -= 14
        elif ptph < -2.5:
            score -= 6

    ptvwap = features.get("price_to_vwap_pct", np.nan)
    if pd.notna(ptvwap):
        if ptvwap >= 0.3:
            score += 8
        elif ptvwap < -0.25:
            score -= 12

    if features.get("vwap_lost_after_open", False):
        score -= 18
    if features.get("opening_range_fail", False):
        score -= 18

    if setup_type == "OPEN_DRIVE_BREAKOUT":
        score += 12
    elif setup_type == "FIRST_PULLBACK":
        score += 8
    elif setup_type == "VWAP_RECLAIM":
        score += 6
    else:
        score -= 20

    return max(0, min(100, int(round(score))))


def compute_intraday_trade_levels(features: dict, setup_type: str) -> dict:
    price = features['last_price']
    vwap = features['session_vwap']
    ema9 = features['ema9']
    atr5m = features['atr5m']
    opening_range_high = features['opening_range_high']
    opening_range_low = features['opening_range_low']
    intraday_low = features['day_low']

    if pd.isna(atr5m) or atr5m <= 0:
        atr5m = max(price * 0.01, 0.05)

    if setup_type == 'OPEN_DRIVE_BREAKOUT':
        trigger = max(x for x in [price, opening_range_high, features['premarket_high']] if pd.notna(x))
        entry = round(trigger * 1.001, 4)
        anchor = max(vwap, opening_range_low) if pd.notna(opening_range_low) else vwap
        stop = round(max(anchor - 0.6 * atr5m, entry * 0.97), 4)
        rr1, rr2 = 1.2, 2.4
    elif setup_type == 'FIRST_PULLBACK':
        anchor_entry = max(x for x in [vwap, ema9, price] if pd.notna(x))
        entry = round(anchor_entry * 1.001, 4)
        anchor = max(x for x in [vwap, intraday_low, opening_range_low] if pd.notna(x))
        stop = round(max(anchor - 0.5 * atr5m, entry * 0.975), 4)
        rr1, rr2 = 1.0, 2.0
    elif setup_type == 'VWAP_RECLAIM':
        entry = round(max(price, vwap) * 1.001, 4)
        stop = round(max(vwap - 0.7 * atr5m, entry * 0.972), 4)
        rr1, rr2 = 1.0, 2.0
    else:
        return {'entry': None, 'stop': None, 'tp1': None, 'tp2': None}

    if stop >= entry:
        stop = round(entry * 0.975, 4)

    risk = max(entry - stop, 0.01)
    tp1 = round(entry + rr1 * risk, 4)
    tp2 = round(entry + rr2 * risk, 4)
    return {'entry': entry, 'stop': stop, 'tp1': tp1, 'tp2': tp2}


@st.cache_data(ttl=30)
def fetch_intraday_trade_universe(session_name: str, max_records: int = 30, min_price: float = 1.0, max_price: float = 100.0):
    if session_name == 'premarket':
        sort_field = 'premarket_change'
        vol_field = 'premarket_volume'
        price_field = 'premarket_close'
        min_vol = 50000
    elif session_name == 'afterhours':
        sort_field = 'postmarket_change'
        vol_field = 'postmarket_volume'
        price_field = 'postmarket_close'
        min_vol = 50000
    else:
        sort_field = 'change'
        vol_field = 'volume'
        price_field = 'close'
        min_vol = 150000

    filters = [
        {'left': price_field, 'operation': 'in_range', 'right': [min_price, max_price]},
        {'left': vol_field, 'operation': 'greater', 'right': min_vol},
        {'left': 'exchange', 'operation': 'in_range', 'right': ['NASDAQ', 'NYSE', 'AMEX']},
    ]

    columns = ['name', 'description', price_field, sort_field, vol_field, 'market_cap_basic']
    raw = tradingview_scan(filters, columns, sort_field, max_records=max_records, page_size=min(100, max_records))

    rows = []
    seen = set()
    for item in raw:
        try:
            d = item['d']
            symbol = d[0]
            description = d[1] or ''
            if not symbol or symbol in seen:
                continue
            if is_likely_non_common_stock(symbol, description):
                continue
            seen.add(symbol)
            rows.append({
                'symbol': symbol,
                'yahoo_symbol': normalize_symbol_for_yahoo(symbol),
                'description': description,
                'live_price': safe_float(d[2], np.nan),
                'live_change_pct': safe_float(d[3], np.nan),
                'live_volume': safe_float(d[4], np.nan),
                'market_cap': safe_float(d[5], np.nan) if len(d) > 5 else np.nan,
            })
        except Exception as exc:
            log_debug(f'Intraday universe parse error: {exc}')
    return pd.DataFrame(rows)


def build_intraday_features(
    symbol: str,
    yahoo_symbol: str,
    daily_df: pd.DataFrame,
    spy_daily_df: pd.DataFrame,
    intraday_df: pd.DataFrame,
    spy_intraday_df: pd.DataFrame | None = None,
) -> dict | None:
    if daily_df is None or daily_df.empty or intraday_df is None or intraday_df.empty:
        return None

    premarket_df, regular_df, afterhours_df = split_sessions(intraday_df)
    min_needed = max(6, _dynamic_min_regular_bars())
    if regular_df.empty or len(regular_df) < min_needed:
        return None

    reg = regular_df.copy()
    reg['EMA9'] = calc_ema(reg['Close'], 9)
    reg['EMA20'] = calc_ema(reg['Close'], 20)
    reg['EMA50'] = calc_ema(reg['Close'], 50)
    macd, macd_signal, macd_hist = calc_macd(reg['Close'])
    reg['MACD'] = macd
    reg['MACD_SIGNAL'] = macd_signal
    reg['MACD_HIST'] = macd_hist
    reg['RSI14'] = calc_rsi(reg['Close'], 14)
    reg['ATR5M'] = calc_intraday_atr(reg, 14)

    last = reg.iloc[-1]
    last_price = float(last['Close'])
    day_high = float(reg['High'].max())
    day_low = float(reg['Low'].min())
    session_vwap = calc_session_vwap(reg)
    if pd.isna(session_vwap) or session_vwap <= 0:
        return None

    # İlk 15 dakika opening range
    first3 = reg.iloc[:3].copy()
    opening_range_high = float(first3['High'].max()) if not first3.empty else np.nan
    opening_range_low = float(first3['Low'].min()) if not first3.empty else np.nan
    opening_range_break = bool(len(reg) > 3 and reg.iloc[3:]['High'].max() > opening_range_high) if pd.notna(opening_range_high) else False

    # Açılış range'i sonra aşağı kırıldı mı? Özellikle long için kötü işaret
    opening_range_fail = False
    if len(reg) > 6 and pd.notna(opening_range_low):
        later_low = float(reg.iloc[3:]['Low'].min())
        opening_range_fail = later_low < opening_range_low * 0.998

    # İlk kontrollü geri çekilme tuttu mu
    pullback_window = reg.iloc[3:9].copy()
    first_pullback_hold = False
    if not pullback_window.empty and pd.notna(opening_range_low):
        first_pullback_hold = float(pullback_window['Low'].min()) >= max(opening_range_low, session_vwap) * 0.985

    # VWAP reclaim: önce altına inip sonra üstüne dönme
    vwap_reclaim = False
    if len(reg) >= 8:
        vwap_reclaim = bool((reg['Close'].iloc[:-1] < session_vwap).any() and last_price > session_vwap)

    # Açılıştan sonra VWAP kaybı ve tekrar altında kalma: long için olumsuz
    vwap_lost_after_open = False
    if len(reg) >= 6:
        after_open = reg.iloc[3:].copy()
        if not after_open.empty:
            vwap_lost_after_open = bool((after_open['Close'] < session_vwap).tail(3).any() and last_price < session_vwap * 1.002)

    prev_close = float(daily_df['Close'].iloc[-2]) if len(daily_df) >= 2 else np.nan
    prev_day_high = float(daily_df['High'].iloc[-2]) if len(daily_df) >= 2 else np.nan
    gap_pct = ((float(reg['Open'].iloc[0]) - prev_close) / prev_close * 100) if prev_close > 0 else np.nan

    avg_vol_20 = daily_df['Volume'].rolling(20).mean().iloc[-1] if len(daily_df) >= 20 else np.nan
    session_volume = float(reg['Volume'].sum())
    # Gün ortasında daha anlamlı olsun diye seans hacmini günlük ortalamaya zamansal normalize et
    bars_so_far = max(len(reg), 1)
    full_day_equiv_volume = session_volume * (78 / bars_so_far)
    session_rvol = (full_day_equiv_volume / avg_vol_20) if pd.notna(avg_vol_20) and avg_vol_20 > 0 else np.nan

    # Intraday RS vs SPY: aynı gün regular seans getirisi ile hesapla
    rs_vs_spy_intraday_pct = 0.0
    if prev_close > 0:
        stock_ret = (last_price - prev_close) / prev_close * 100
        spy_ret = 0.0
        if spy_intraday_df is not None and not spy_intraday_df.empty:
            _, spy_reg, _ = split_sessions(spy_intraday_df)
            if not spy_reg.empty:
                spy_prev_close = float(spy_daily_df['Close'].iloc[-2]) if spy_daily_df is not None and not spy_daily_df.empty and len(spy_daily_df) >= 2 else np.nan
                spy_last = float(spy_reg['Close'].iloc[-1])
                if pd.notna(spy_prev_close) and spy_prev_close > 0:
                    spy_ret = (spy_last - spy_prev_close) / spy_prev_close * 100
        elif spy_daily_df is not None and not spy_daily_df.empty and len(spy_daily_df) >= 2:
            spy_prev_close = float(spy_daily_df['Close'].iloc[-2])
            spy_last = float(spy_daily_df['Close'].iloc[-1])
            if spy_prev_close > 0:
                spy_ret = (spy_last - spy_prev_close) / spy_prev_close * 100
        rs_vs_spy_intraday_pct = stock_ret - spy_ret

    price_to_day_high_pct = ((last_price - day_high) / day_high * 100) if day_high > 0 else np.nan
    price_to_vwap_pct = ((last_price - session_vwap) / session_vwap * 100) if session_vwap > 0 else np.nan
    ema9 = float(last['EMA9']) if pd.notna(last['EMA9']) else np.nan
    ema20 = float(last['EMA20']) if pd.notna(last['EMA20']) else np.nan
    ema50 = float(last['EMA50']) if pd.notna(last['EMA50']) else np.nan
    price_to_ema9_pct = ((last_price - ema9) / ema9 * 100) if pd.notna(ema9) and ema9 > 0 else np.nan

    premarket_high = float(premarket_df['High'].max()) if not premarket_df.empty else np.nan
    premarket_low = float(premarket_df['Low'].min()) if not premarket_df.empty else np.nan
    premarket_volume = float(premarket_df['Volume'].sum()) if not premarket_df.empty else 0.0
    price_to_premarket_high_pct = ((last_price - premarket_high) / premarket_high * 100) if pd.notna(premarket_high) and premarket_high > 0 else np.nan

    return {
        'last_price': last_price,
        'session_vwap': session_vwap,
        'ema9': ema9,
        'ema20': ema20,
        'ema50': ema50,
        'ema9_above_ema20': bool(pd.notna(ema9) and pd.notna(ema20) and ema9 > ema20),
        'ema20_above_ema50': bool(pd.notna(ema20) and pd.notna(ema50) and ema20 > ema50),
        'macd': float(last['MACD']) if pd.notna(last['MACD']) else np.nan,
        'macd_signal': float(last['MACD_SIGNAL']) if pd.notna(last['MACD_SIGNAL']) else np.nan,
        'macd_hist': float(last['MACD_HIST']) if pd.notna(last['MACD_HIST']) else np.nan,
        'macd_hist_positive': bool(pd.notna(last['MACD_HIST']) and last['MACD_HIST'] > 0),
        'rsi14': float(last['RSI14']) if pd.notna(last['RSI14']) else np.nan,
        'atr5m': float(last['ATR5M']) if pd.notna(last['ATR5M']) else np.nan,
        'close_above_vwap': bool(last_price > session_vwap),
        'opening_range_high': opening_range_high,
        'opening_range_low': opening_range_low,
        'opening_range_break': opening_range_break,
        'opening_range_fail': opening_range_fail,
        'first_pullback_hold': first_pullback_hold,
        'vwap_reclaim': vwap_reclaim,
        'vwap_lost_after_open': vwap_lost_after_open,
        'day_high': day_high,
        'day_low': day_low,
        'premarket_high': premarket_high,
        'premarket_low': premarket_low,
        'premarket_volume': premarket_volume,
        'gap_pct': gap_pct,
        'session_volume': session_volume,
        'session_rvol': session_rvol,
        'price_to_day_high_pct': price_to_day_high_pct,
        'price_to_premarket_high_pct': price_to_premarket_high_pct,
        'price_to_vwap_pct': price_to_vwap_pct,
        'price_to_ema9_pct': price_to_ema9_pct,
        'prev_day_high': prev_day_high,
        'rs_vs_spy_intraday_pct': rs_vs_spy_intraday_pct,
    }


def evaluate_intraday_candidates(universe_df: pd.DataFrame, daily_dict: dict, spy_daily_df: pd.DataFrame):
    final_candidates = []
    rejected = []

    spy_intraday_df, spy_intraday_reason = get_intraday_session_data_verbose(
        symbol='SPY',
        yahoo_symbol='SPY',
        api_key_value=api_key,
        secret_key_value=secret_key,
        alpaca_feed=os.getenv('ALPACA_FEED', 'iex'),
    )
    if spy_intraday_df.empty:
        log_debug(f"SPY intraday fallback -> {spy_intraday_reason}")

    for _, row in universe_df.iterrows():
        symbol = row['symbol']
        yahoo_symbol = row['yahoo_symbol']
        daily_df = daily_dict.get(yahoo_symbol)
        if daily_df is None or daily_df.empty or len(daily_df) < 60:
            rejected.append({'Hisse': symbol, 'Neden': 'Yetersiz günlük veri'})
            continue

        intraday_df, intraday_reason = get_intraday_session_data_verbose(
            symbol=symbol,
            yahoo_symbol=yahoo_symbol,
            api_key_value=api_key,
            secret_key_value=secret_key,
            alpaca_feed=os.getenv('ALPACA_FEED', 'iex'),
        )
        if intraday_df.empty:
            rejected.append({'Hisse': symbol, 'Neden': intraday_reason})
            continue

        feats = build_intraday_features(symbol, yahoo_symbol, daily_df, spy_daily_df, intraday_df, spy_intraday_df=spy_intraday_df)
        if feats is None:
            rejected.append({'Hisse': symbol, 'Neden': 'İntraday feature üretilemedi'})
            continue

        # Global hard reject: yumuşak ve bozulmuş adayları en başta at
        if pd.isna(feats['session_rvol']) or feats['session_rvol'] < 1.2:
            rejected.append({'Hisse': symbol, 'Neden': f"Session RVOL düşük ({round(feats['session_rvol'], 2) if pd.notna(feats['session_rvol']) else 'NaN'})"})
            continue
        if pd.notna(feats['price_to_day_high_pct']) and feats['price_to_day_high_pct'] < -2.0:
            rejected.append({'Hisse': symbol, 'Neden': f"Day-highdan fazla uzak ({round(feats['price_to_day_high_pct'], 2)}%)"})
            continue
        if pd.notna(feats['rsi14']) and feats['rsi14'] < 55:
            rejected.append({'Hisse': symbol, 'Neden': f"RSI zayıf ({round(feats['rsi14'], 2)})"})
            continue
        if feats.get('opening_range_fail', False):
            rejected.append({'Hisse': symbol, 'Neden': 'Opening range fail'})
            continue
        if feats.get('vwap_lost_after_open', False):
            rejected.append({'Hisse': symbol, 'Neden': 'VWAP kaybı'})
            continue
        if pd.notna(feats.get('price_to_premarket_high_pct', np.nan)) and feats['price_to_premarket_high_pct'] < -4.0:
            rejected.append({'Hisse': symbol, 'Neden': f"Premarket highdan koptu ({round(feats['price_to_premarket_high_pct'], 2)}%)"})
            continue

        setup_type = detect_intraday_setup(feats)
        if setup_type == 'NO_SETUP':
            rejected.append({'Hisse': symbol, 'Neden': 'Net intraday setup yok'})
            continue

        confidence = intraday_confidence_score(feats, setup_type)
        if confidence < 72:
            rejected.append({'Hisse': symbol, 'Neden': f'Confidence düşük ({confidence})'})
            continue

        levels = compute_intraday_trade_levels(feats, setup_type)
        if levels['entry'] is None or levels['stop'] is None:
            rejected.append({'Hisse': symbol, 'Neden': 'Trade seviyeleri hesaplanamadı'})
            continue

        risk = max(levels['entry'] - levels['stop'], 0.01)
        pos = calc_position_size(
            account_size=2000.0,
            risk_per_trade_pct=0.02,
            entry=levels['entry'],
            stop=levels['stop'],
        )

        why_parts = []
        if feats['opening_range_break']:
            why_parts.append('ORB')
        if feats['first_pullback_hold']:
            why_parts.append('İlk pullback tuttu')
        if feats['close_above_vwap']:
            why_parts.append('VWAP üstü')
        if feats['ema9_above_ema20']:
            why_parts.append('EMA9>EMA20')
        if feats['macd_hist_positive']:
            why_parts.append('MACD+')
        if pd.notna(feats['price_to_premarket_high_pct']) and feats['price_to_premarket_high_pct'] >= -1.5:
            why_parts.append('PM high yakınında')

        final_candidates.append({
            'Symbol': symbol,
            'Setup_Type': setup_type,
            'Confidence': confidence,
            'Price': round(feats['last_price'], 4 if feats['last_price'] < 1 else 2),
            'Gap_%': round(feats['gap_pct'], 2) if pd.notna(feats['gap_pct']) else np.nan,
            'Session_RVOL': round(feats['session_rvol'], 2) if pd.notna(feats['session_rvol']) else np.nan,
            'RSI14': round(feats['rsi14'], 2) if pd.notna(feats['rsi14']) else np.nan,
            'MACD_Hist': round(feats['macd_hist'], 4) if pd.notna(feats['macd_hist']) else np.nan,
            'VWAP': round(feats['session_vwap'], 4 if feats['session_vwap'] < 1 else 2),
            'EMA9': round(feats['ema9'], 4 if pd.notna(feats['ema9']) and feats['ema9'] < 1 else 2) if pd.notna(feats['ema9']) else np.nan,
            'EMA20': round(feats['ema20'], 4 if pd.notna(feats['ema20']) and feats['ema20'] < 1 else 2) if pd.notna(feats['ema20']) else np.nan,
            'Price_to_DayHigh_%': round(feats['price_to_day_high_pct'], 2) if pd.notna(feats['price_to_day_high_pct']) else np.nan,
            'Price_to_PMHigh_%': round(feats['price_to_premarket_high_pct'], 2) if pd.notna(feats['price_to_premarket_high_pct']) else np.nan,
            'Price_to_VWAP_%': round(feats['price_to_vwap_pct'], 2) if pd.notna(feats['price_to_vwap_pct']) else np.nan,
            'RS_vs_SPY_%': round(feats['rs_vs_spy_intraday_pct'], 2),
            'Entry': levels['entry'],
            'Stop': levels['stop'],
            'TP1': levels['tp1'],
            'TP2': levels['tp2'],
            'Risk_per_Share': round(risk, 4),
            'Model_Why': ', '.join(why_parts[:6]),
            'Suggested_Shares_2k_2pct': pos['shares'],
        })

    final_candidates = sorted(
        final_candidates,
        key=lambda x: (
            x['Confidence'],
            x['Session_RVOL'] if not pd.isna(x['Session_RVOL']) else 0,
            x['Price_to_DayHigh_%'] if not pd.isna(x['Price_to_DayHigh_%']) else -999,
        ),
        reverse=True
    )
    return final_candidates, rejected


# ============================================================
# NIGHT BUY / OVERNIGHT PRESSURE ENGINE
# ============================================================
def _score_linear(value, low, high, clamp=True):
    try:
        if pd.isna(value):
            return 0.0
        score = (float(value) - low) / (high - low) * 100.0
        if clamp:
            return float(max(0.0, min(100.0, score)))
        return float(score)
    except Exception:
        return 0.0


def _rsi_strength_score(rsi: float) -> float:
    if pd.isna(rsi):
        return 0.0
    # Night-buy için ideal bölge: 55-75. 80+ momentum olabilir ama fakeout riski de artar.
    if 55 <= rsi <= 68:
        return 100.0
    if 68 < rsi <= 75:
        return 85.0
    if 50 <= rsi < 55:
        return 55.0
    if 75 < rsi <= 82:
        return 55.0
    if rsi > 82:
        return 20.0
    return 10.0


def _safe_pct(numerator: float, denominator: float) -> float:
    try:
        if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
            return np.nan
        return (numerator - denominator) / denominator * 100.0
    except Exception:
        return np.nan


def _target_pct_by_price(price: float) -> float:
    if pd.isna(price) or price <= 0:
        return 0.08
    if price < 3:
        return 0.15
    if price < 10:
        return 0.10
    if price < 50:
        return 0.06
    return 0.035


def _price_round(price: float) -> int:
    if pd.isna(price):
        return 4
    return 4 if price < 1 else 3 if price < 10 else 2


def calc_daily_indicators_for_night(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["EMA9"] = calc_ema(temp["Close"], 9)
    temp["EMA20"] = calc_ema(temp["Close"], 20)
    temp["EMA50"] = calc_ema(temp["Close"], 50)
    temp["RSI14"] = calc_rsi(temp["Close"], 14)
    macd, macd_signal, macd_hist = calc_macd(temp["Close"])
    temp["MACD"] = macd
    temp["MACD_SIGNAL"] = macd_signal
    temp["MACD_HIST"] = macd_hist
    temp["ATR14"] = calc_atr(temp, 14)
    return temp


@st.cache_data(ttl=30)
def fetch_night_buy_universe(
    scan_mode: str = "afterhours",
    max_records: int = 80,
    min_price: float = 1.0,
    max_price: float = 80.0,
    min_volume: int = 100000,
) -> pd.DataFrame:
    """TradingView üzerinden night-buy evrenini alır.
    Bilinçli olarak sadece daha stabil TradingView kolonları kullanıldı; short/float verisi bu sürümde proxy skorla hesaplanır.
    """
    if scan_mode == "afterhours":
        sort_field = "postmarket_change"
        vol_field = "postmarket_volume"
        price_field = "postmarket_close"
        min_live_vol = max(10000, int(min_volume * 0.08))
    elif scan_mode == "premarket":
        sort_field = "premarket_change"
        vol_field = "premarket_volume"
        price_field = "premarket_close"
        min_live_vol = max(10000, int(min_volume * 0.08))
    else:
        sort_field = "relative_volume_10d_calc"
        vol_field = "volume"
        price_field = "close"
        min_live_vol = min_volume

    filters = [
        {"left": price_field, "operation": "in_range", "right": [min_price, max_price]},
        {"left": vol_field, "operation": "greater", "right": min_live_vol},
        {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
    ]

    columns = [
        "name",
        "description",
        price_field,
        sort_field,
        vol_field,
        "close",
        "volume",
        "relative_volume_10d_calc",
        "gap",
        "market_cap_basic",
    ]

    raw = tradingview_scan(filters, columns, sort_field, max_records=max_records, page_size=min(100, max_records))
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
            rows.append({
                "symbol": symbol,
                "yahoo_symbol": normalize_symbol_for_yahoo(symbol),
                "description": description,
                "live_price": safe_float(d[2], np.nan),
                "live_change_pct": safe_float(d[3], np.nan),
                "live_volume": safe_float(d[4], np.nan),
                "regular_close": safe_float(d[5], np.nan),
                "regular_volume": safe_float(d[6], np.nan),
                "tv_rvol": safe_float(d[7], np.nan),
                "tv_gap": safe_float(d[8], np.nan),
                "market_cap": safe_float(d[9], np.nan),
            })
        except Exception as exc:
            log_debug(f"Night universe parse error: {exc}")
    return pd.DataFrame(rows)


def get_recent_intraday_full_data(
    symbol: str,
    yahoo_symbol: str,
    api_key_value: str = "",
    secret_key_value: str = "",
    alpaca_feed: str = "iex",
) -> pd.DataFrame:
    """After-hours/premarket dahil son aktif günün intraday datası.
    Mevcut get_intraday_session_data regular session seçmeye odaklandığı için night-buy için ayrı tutuldu.
    """
    intraday = fetch_alpaca_intraday_5m(symbol, api_key_value, secret_key_value, alpaca_feed)
    if intraday is not None and not intraday.empty:
        picked = _extract_last_active_day(intraday)
        if picked is not None and not picked.empty:
            return picked

    yahoo_candidates = _fetch_yahoo_intraday_candidates(yahoo_symbol)
    for key in ["1m_2d", "5m_5d"]:
        cand = yahoo_candidates.get(key, pd.DataFrame())
        if cand is None or cand.empty:
            continue
        picked = _extract_last_active_day(cand)
        if picked is not None and not picked.empty:
            return picked
    return pd.DataFrame()


def compute_fib_channel_features(df: pd.DataFrame, price: float) -> dict:
    if df is None or df.empty or len(df) < 40 or pd.isna(price) or price <= 0:
        return {
            "swing_low": np.nan,
            "swing_high": np.nan,
            "fib_1272": np.nan,
            "fib_1618": np.nan,
            "channel_score": 0.0,
            "upside_to_1272_pct": np.nan,
        }
    recent = df.tail(60).copy()
    swing_low = float(recent["Low"].min())
    swing_high = float(recent["High"].max())
    wave = swing_high - swing_low
    if wave <= 0:
        return {
            "swing_low": swing_low,
            "swing_high": swing_high,
            "fib_1272": np.nan,
            "fib_1618": np.nan,
            "channel_score": 0.0,
            "upside_to_1272_pct": np.nan,
        }

    fib_1272 = swing_low + 1.272 * wave
    fib_1618 = swing_low + 1.618 * wave
    upside_1272 = (fib_1272 - price) / price * 100 if price > 0 else np.nan

    prior_20h = float(df["High"].shift(1).rolling(20).max().iloc[-1]) if len(df) >= 21 else np.nan
    prior_60h = float(df["High"].shift(1).rolling(60).max().iloc[-1]) if len(df) >= 61 else np.nan

    score = 0.0
    if pd.notna(prior_20h):
        if price >= prior_20h * 0.995:
            score += 30
        elif price >= prior_20h * 0.97:
            score += 18
    if pd.notna(prior_60h):
        if price >= prior_60h * 0.98:
            score += 20
        elif price >= prior_60h * 0.94:
            score += 10
    if pd.notna(upside_1272):
        if upside_1272 >= 12:
            score += 35
        elif upside_1272 >= 7:
            score += 25
        elif upside_1272 >= 3:
            score += 12
        elif upside_1272 < 0:
            score -= 15
    # Fiyat son dalganın çok tepesindeyse ama extension alanı kalmadıysa fakeout riski artar.
    if pd.notna(fib_1618) and price > fib_1618:
        score -= 18

    return {
        "swing_low": swing_low,
        "swing_high": swing_high,
        "fib_1272": fib_1272,
        "fib_1618": fib_1618,
        "channel_score": float(max(0, min(100, score))),
        "upside_to_1272_pct": upside_1272,
    }


def compute_night_buy_candidate(row: pd.Series, daily_df: pd.DataFrame, intraday_df: pd.DataFrame) -> tuple[dict | None, str | None]:
    symbol = row["symbol"]
    yahoo_symbol = row["yahoo_symbol"]
    if daily_df is None or daily_df.empty or len(daily_df) < 80:
        return None, "Yetersiz günlük veri"

    dfi = calc_daily_indicators_for_night(daily_df).dropna(subset=["Close", "Volume"]).copy()
    if dfi.empty or len(dfi) < 60:
        return None, "İndikatör için yetersiz veri"

    last = dfi.iloc[-1]
    prev = dfi.iloc[-2]
    last_close = float(last["Close"])
    last_open = float(last["Open"])
    last_high = float(last["High"])
    last_low = float(last["Low"])
    last_volume = float(last["Volume"])
    live_price = safe_float(row.get("live_price", np.nan), np.nan)
    if pd.isna(live_price) or live_price <= 0:
        live_price = last_close

    atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else max(last_close * 0.04, 0.05)
    avg_vol_20 = float(dfi["Volume"].rolling(20).mean().iloc[-1])
    rvol20 = last_volume / avg_vol_20 if avg_vol_20 > 0 else np.nan
    if pd.notna(row.get("tv_rvol", np.nan)) and row.get("tv_rvol", np.nan) > 0:
        rvol20 = max(rvol20 if pd.notna(rvol20) else 0, float(row["tv_rvol"]))

    closing_strength = calc_closing_strength(last_close, last_low, last_high)
    prev_day_change_pct = (last_close - float(prev["Close"])) / float(prev["Close"]) * 100 if float(prev["Close"]) > 0 else np.nan

    ema9 = float(last["EMA9"]) if pd.notna(last["EMA9"]) else np.nan
    ema20 = float(last["EMA20"]) if pd.notna(last["EMA20"]) else np.nan
    ema50 = float(last["EMA50"]) if pd.notna(last["EMA50"]) else np.nan
    ema9_slope = float(last["EMA9"] - dfi["EMA9"].iloc[-4]) if len(dfi) >= 4 and pd.notna(dfi["EMA9"].iloc[-4]) else np.nan
    ema20_slope = float(last["EMA20"] - dfi["EMA20"].iloc[-4]) if len(dfi) >= 4 and pd.notna(dfi["EMA20"].iloc[-4]) else np.nan
    rsi14 = float(last["RSI14"]) if pd.notna(last["RSI14"]) else np.nan
    macd_hist = float(last["MACD_HIST"]) if pd.notna(last["MACD_HIST"]) else np.nan
    macd_hist_prev = float(dfi["MACD_HIST"].iloc[-2]) if pd.notna(dfi["MACD_HIST"].iloc[-2]) else np.nan
    macd_rising = pd.notna(macd_hist) and pd.notna(macd_hist_prev) and macd_hist > macd_hist_prev

    pre_df, reg_df, ah_df = split_sessions(intraday_df) if intraday_df is not None and not intraday_df.empty else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    reg_vwap = calc_session_vwap(reg_df) if reg_df is not None and not reg_df.empty else np.nan
    ah_vwap = calc_session_vwap(ah_df) if ah_df is not None and not ah_df.empty else np.nan
    ah_price = float(ah_df["Close"].iloc[-1]) if ah_df is not None and not ah_df.empty else live_price
    ah_high = float(ah_df["High"].max()) if ah_df is not None and not ah_df.empty else np.nan
    ah_low = float(ah_df["Low"].min()) if ah_df is not None and not ah_df.empty else np.nan
    ah_volume = float(ah_df["Volume"].sum()) if ah_df is not None and not ah_df.empty else safe_float(row.get("live_volume", 0), 0)
    ah_change_pct = _safe_pct(ah_price, last_close)
    live_change_pct = safe_float(row.get("live_change_pct", np.nan), np.nan)
    if pd.isna(ah_change_pct) and pd.notna(live_change_pct):
        ah_change_pct = live_change_pct

    price_to_ema9_pct = _safe_pct(ah_price, ema9) if pd.notna(ema9) else np.nan
    ah_volume_to_daily_pct = (ah_volume / avg_vol_20 * 100) if avg_vol_20 > 0 else np.nan

    # Talep basıncı: hacim + kapanış gücü + AH hacim/fiyat devamı.
    demand_parts = []
    demand_parts.append(_score_linear(rvol20, 1.0, 6.0) * 0.40)
    demand_parts.append((closing_strength if pd.notna(closing_strength) else 0) * 100 * 0.22)
    demand_parts.append(_score_linear(ah_volume_to_daily_pct, 1.0, 12.0) * 0.18)
    demand_parts.append(_score_linear(ah_change_pct, 0.0, 8.0) * 0.20)
    demand_score = float(max(0, min(100, sum(demand_parts))))

    # Teknik uyum: EMA9>EMA20, EMA eğimleri, MACD ivmesi, RSI verimli güç bölgesi, VWAP üstü.
    technical_score = 0.0
    if pd.notna(ema9) and pd.notna(ema20) and ema9 > ema20:
        technical_score += 22
    if pd.notna(ema20) and pd.notna(ema50) and ema20 > ema50:
        technical_score += 10
    if pd.notna(ema9) and ah_price > ema9:
        technical_score += 14
    if pd.notna(ema9_slope) and ema9_slope > 0:
        technical_score += 10
    if pd.notna(ema20_slope) and ema20_slope >= 0:
        technical_score += 6
    if pd.notna(macd_hist) and macd_hist > 0:
        technical_score += 12
    if macd_rising:
        technical_score += 10
    technical_score += _rsi_strength_score(rsi14) * 0.16
    if pd.notna(reg_vwap) and ah_price > reg_vwap:
        technical_score += 8
    if pd.notna(ah_vwap) and ah_price > ah_vwap:
        technical_score += 8
    technical_score = float(max(0, min(100, technical_score)))

    # Squeeze katkısı: bu sürümde gerçek short-float yoksa proxy kullanır.
    # Düşük piyasa değeri + yüksek RVOL + yüksek hacim + fiyatın kırılım bölgesinde olması squeeze basıncı olarak yorumlanır.
    market_cap = safe_float(row.get("market_cap", np.nan), np.nan)
    squeeze_score = 0.0
    if pd.notna(market_cap):
        if market_cap < 75_000_000:
            squeeze_score += 28
        elif market_cap < 300_000_000:
            squeeze_score += 22
        elif market_cap < 1_000_000_000:
            squeeze_score += 12
    if pd.notna(rvol20):
        if rvol20 >= 8:
            squeeze_score += 28
        elif rvol20 >= 5:
            squeeze_score += 22
        elif rvol20 >= 3:
            squeeze_score += 14
    if pd.notna(ah_volume_to_daily_pct):
        if ah_volume_to_daily_pct >= 12:
            squeeze_score += 18
        elif ah_volume_to_daily_pct >= 5:
            squeeze_score += 12
        elif ah_volume_to_daily_pct >= 2:
            squeeze_score += 6
    prior_20h = float(dfi["High"].shift(1).rolling(20).max().iloc[-1]) if len(dfi) >= 21 else np.nan
    extension_above_20h_pct = _safe_pct(ah_price, prior_20h) if pd.notna(prior_20h) else np.nan
    if pd.notna(prior_20h) and ah_price >= prior_20h * 0.995:
        squeeze_score += 18
    if pd.notna(live_change_pct) and live_change_pct >= 8:
        squeeze_score += 8
    squeeze_score = float(max(0, min(100, squeeze_score)))

    fib = compute_fib_channel_features(dfi, ah_price)
    channel_score = fib["channel_score"]

    # After-hours strength: stratejinin kendisi night-buy olduğu için ayrı güç skoru.
    ah_strength = 0.0
    if pd.notna(ah_change_pct):
        if ah_change_pct >= 8:
            ah_strength += 30
        elif ah_change_pct >= 4:
            ah_strength += 22
        elif ah_change_pct >= 1.5:
            ah_strength += 12
        elif ah_change_pct < -1:
            ah_strength -= 12
    if pd.notna(ah_vwap) and ah_price > ah_vwap:
        ah_strength += 26
    if pd.notna(reg_vwap) and ah_price > reg_vwap:
        ah_strength += 16
    if pd.notna(ah_volume_to_daily_pct):
        if ah_volume_to_daily_pct >= 10:
            ah_strength += 22
        elif ah_volume_to_daily_pct >= 4:
            ah_strength += 14
        elif ah_volume_to_daily_pct >= 1:
            ah_strength += 6
    if pd.notna(ah_high) and ah_high > 0:
        ah_close_strength = (ah_price - ah_low) / (ah_high - ah_low) if pd.notna(ah_low) and ah_high > ah_low else np.nan
        if pd.notna(ah_close_strength):
            if ah_close_strength >= 0.75:
                ah_strength += 10
            elif ah_close_strength < 0.35:
                ah_strength -= 10
    ah_strength = float(max(0, min(100, ah_strength)))

    # Piyasa/sektör desteği: burada SPY canlı intraday entegrasyonu yok; nötr 50 veriyoruz.
    market_support = 50.0

    # Fakeout/risk: üst fitil, aşırı uzama, EMA9'dan kopma, RSI şişmesi, AH VWAP altı, önceki gün fazla prim.
    fakeout_risk = 0.0
    if pd.notna(closing_strength):
        if closing_strength < 0.45:
            fakeout_risk += 22
        elif closing_strength < 0.65:
            fakeout_risk += 12
    if pd.notna(prev_day_change_pct):
        if prev_day_change_pct >= 25:
            fakeout_risk += 24
        elif prev_day_change_pct >= 15:
            fakeout_risk += 16
        elif prev_day_change_pct >= 10:
            fakeout_risk += 10
    if pd.notna(price_to_ema9_pct):
        if price_to_ema9_pct >= 18:
            fakeout_risk += 24
        elif price_to_ema9_pct >= 10:
            fakeout_risk += 14
        elif price_to_ema9_pct <= -2:
            fakeout_risk += 12
    if pd.notna(rsi14):
        if rsi14 >= 85:
            fakeout_risk += 22
        elif rsi14 >= 78:
            fakeout_risk += 12
        elif rsi14 < 50:
            fakeout_risk += 18
    if pd.notna(ah_vwap) and ah_price < ah_vwap:
        fakeout_risk += 20
    if pd.notna(extension_above_20h_pct):
        if extension_above_20h_pct >= 18:
            fakeout_risk += 20
        elif extension_above_20h_pct >= 10:
            fakeout_risk += 10
    if pd.notna(ah_volume_to_daily_pct) and ah_volume_to_daily_pct < 0.5:
        fakeout_risk += 12
    fakeout_risk = float(max(0, min(100, fakeout_risk)))

    positive_score = (
        0.25 * demand_score +
        0.20 * technical_score +
        0.20 * squeeze_score +
        0.15 * ah_strength +
        0.15 * channel_score +
        0.05 * market_support
    )
    # Risk doğrudan çıkarılır; bu yüzden iyi görünen ama fakeout riski yüksek adaylar elenir.
    final_score = float(max(0, min(100, positive_score - 0.55 * fakeout_risk)))

    hard_reject_reasons = []
    if pd.isna(rvol20) or rvol20 < 1.3:
        hard_reject_reasons.append("RVOL düşük")
    if technical_score < 45:
        hard_reject_reasons.append("Teknik hizalanma zayıf")
    if ah_strength < 30:
        hard_reject_reasons.append("After-hours güç zayıf")
    if fakeout_risk >= 70:
        hard_reject_reasons.append("Fakeout riski çok yüksek")
    if ah_price < 1:
        hard_reject_reasons.append("Fiyat < 1$")

    target_pct = _target_pct_by_price(ah_price)
    # Giriş bölgesi: AH VWAP ve son fiyatın üzerinde kontrollü bölge. Tek fiyat yerine bölge verilir.
    entry_base_candidates = [ah_price]
    if pd.notna(ah_vwap) and ah_vwap > 0:
        entry_base_candidates.append(ah_vwap * 1.002)
    if pd.notna(reg_vwap) and reg_vwap > 0:
        entry_base_candidates.append(reg_vwap * 1.002)
    entry_low = max(entry_base_candidates)
    entry_high = entry_low * (1.012 if ah_price < 10 else 1.006)

    stop_anchor = ah_vwap if pd.notna(ah_vwap) and ah_vwap > 0 else reg_vwap if pd.notna(reg_vwap) and reg_vwap > 0 else last_close
    stop_price = min(entry_low * 0.94, stop_anchor - 0.65 * atr14)
    stop_price = max(0.01, stop_price)
    if stop_price >= entry_low:
        stop_price = entry_low * 0.94
    risk = max(entry_low - stop_price, 0.01)
    tp1 = max(entry_low + 1.2 * risk, entry_low * (1 + target_pct))
    tp2 = max(entry_low + 2.4 * risk, entry_low * (1 + 1.75 * target_pct))

    # Karar katmanları:
    # - A/B+ gerçek işlem adayı değildir; önce küçük/paper test ile doğrulanmalıdır.
    # - Paper Watchlist, 60-74 arası skoru kullanıcıya görünür kılar; sistem eğitim verisi toplamak içindir.
    # - Aggressive Squeeze Watch, final skoru düşük kalsa bile squeeze/talep/AH basıncı yüksek olan riskli adayları ayrı gösterir.
    # - Hard reject varsa aday ana/watchlist/aggressive tablosuna alınmaz, ancak reject bölümünde gerekçesi görünür.
    aggressive_reasons = []
    if squeeze_score >= 65:
        aggressive_reasons.append("Squeeze proxy yüksek")
    if demand_score >= 70:
        aggressive_reasons.append("Talep basıncı yüksek")
    if ah_strength >= 65 and technical_score >= 80 and demand_score >= 60:
        aggressive_reasons.append("AH + teknik momentum güçlü")

    aggressive_watch_ok = (
        final_score < 60
        and fakeout_risk <= 60
        and not hard_reject_reasons
        and (
            squeeze_score >= 65
            or demand_score >= 70
            or (ah_strength >= 65 and technical_score >= 80 and demand_score >= 60)
        )
    )

    if final_score >= 85 and fakeout_risk <= 45 and not hard_reject_reasons:
        grade = "A"
        status = "TRADE_CANDIDATE"
    elif final_score >= 75 and fakeout_risk <= 55 and not hard_reject_reasons:
        grade = "B+"
        status = "TRADE_CANDIDATE"
    elif final_score >= 60 and fakeout_risk <= 60 and not hard_reject_reasons:
        grade = "Paper Watchlist"
        status = "PAPER_WATCHLIST"
    elif aggressive_watch_ok:
        grade = "Aggressive Squeeze Watch"
        status = "AGGRESSIVE_SQUEEZE_WATCH"
    else:
        grade = "Reject / Watch Only"
        status = "REJECT"

    why = []
    if demand_score >= 70:
        why.append("Talep/hacim güçlü")
    if technical_score >= 70:
        why.append("EMA/MACD/RSI hizalı")
    if squeeze_score >= 60:
        why.append("Squeeze proxy güçlü")
    if ah_strength >= 65:
        why.append("AH tutunma güçlü")
    if channel_score >= 60:
        why.append("Kanal/Fib alanı açık")

    risk_notes = []
    if fakeout_risk >= 50:
        risk_notes.append("Fakeout riski yüksek")
    if pd.notna(prev_day_change_pct) and prev_day_change_pct >= 10:
        risk_notes.append("Önceki gün fazla prim")
    if pd.notna(price_to_ema9_pct) and price_to_ema9_pct >= 10:
        risk_notes.append("EMA9'dan uzak")
    if pd.notna(rsi14) and rsi14 >= 78:
        risk_notes.append("RSI şişmiş")
    if hard_reject_reasons:
        risk_notes.extend(hard_reject_reasons[:3])

    rnd = _price_round(ah_price)
    result = {
        "Symbol": symbol,
        "Description": row.get("description", ""),
        "Grade": grade,
        "Status": status,
        "Final_Night_Score": round(final_score, 1),
        "Demand_Pressure": round(demand_score, 1),
        "Technical_Alignment": round(technical_score, 1),
        "Squeeze_Proxy": round(squeeze_score, 1),
        "AH_Strength": round(ah_strength, 1),
        "Channel_Clarity": round(channel_score, 1),
        "Fakeout_Risk": round(fakeout_risk, 1),
        "Last_Close": round(last_close, rnd),
        "AH_Live_Price": round(ah_price, rnd),
        "AH_Change_%": round(ah_change_pct, 2) if pd.notna(ah_change_pct) else np.nan,
        "RVOL20": round(rvol20, 2) if pd.notna(rvol20) else np.nan,
        "AH_Vol_to_Daily_%": round(ah_volume_to_daily_pct, 2) if pd.notna(ah_volume_to_daily_pct) else np.nan,
        "EMA9": round(ema9, rnd) if pd.notna(ema9) else np.nan,
        "EMA20": round(ema20, rnd) if pd.notna(ema20) else np.nan,
        "EMA9_gt_EMA20": bool(pd.notna(ema9) and pd.notna(ema20) and ema9 > ema20),
        "RSI14": round(rsi14, 2) if pd.notna(rsi14) else np.nan,
        "MACD_Hist": round(macd_hist, 4) if pd.notna(macd_hist) else np.nan,
        "MACD_Rising": bool(macd_rising),
        "AH_VWAP": round(ah_vwap, rnd) if pd.notna(ah_vwap) else np.nan,
        "Regular_VWAP": round(reg_vwap, rnd) if pd.notna(reg_vwap) else np.nan,
        "Fib_1272": round(fib["fib_1272"], rnd) if pd.notna(fib["fib_1272"]) else np.nan,
        "Upside_to_Fib127_%": round(fib["upside_to_1272_pct"], 2) if pd.notna(fib["upside_to_1272_pct"]) else np.nan,
        "Night_Entry_Low": round(entry_low, rnd),
        "Night_Entry_High": round(entry_high, rnd),
        "Stop": round(stop_price, rnd),
        "TP1": round(tp1, rnd),
        "TP2": round(tp2, rnd),
        "Risk_per_Share": round(risk, rnd),
        "Why": ", ".join(why[:5]) if why else "Net pozitif gerekçe zayıf",
        "Risk_Notes": ", ".join(risk_notes[:6]) if risk_notes else "Belirgin kırmızı bayrak yok",
        "Aggressive_Reason": ", ".join(aggressive_reasons[:4]) if aggressive_reasons else "-",
        "Hard_Reject": bool(len(hard_reject_reasons) > 0),
    }
    return result, None


def evaluate_night_buy_candidates(universe_df: pd.DataFrame, daily_dict: dict) -> tuple[list[dict], list[dict]]:
    final_candidates = []
    rejected = []
    if universe_df is None or universe_df.empty:
        return final_candidates, rejected

    for _, row in universe_df.iterrows():
        symbol = row["symbol"]
        yahoo_symbol = row["yahoo_symbol"]
        try:
            daily_df = daily_dict.get(yahoo_symbol)
            if daily_df is None or daily_df.empty:
                rejected.append({"Hisse": symbol, "Neden": "Günlük veri yok"})
                continue
            intraday_df = get_recent_intraday_full_data(
                symbol=symbol,
                yahoo_symbol=yahoo_symbol,
                api_key_value=api_key,
                secret_key_value=secret_key,
                alpaca_feed=os.getenv("ALPACA_FEED", "iex"),
            )
            candidate, reason = compute_night_buy_candidate(row, daily_df, intraday_df)
            if candidate is None:
                rejected.append({"Hisse": symbol, "Neden": reason or "Night feature üretilemedi"})
                continue
            # Tüm skorlanmış adayları ana listeye ekliyoruz.
            # UI tarafında bunlar TRADE_CANDIDATE / PAPER_WATCHLIST / REJECT olarak ayrılır.
            final_candidates.append(candidate)
        except Exception as exc:
            rejected.append({"Hisse": symbol, "Neden": f"Hata: {exc}"})
            log_debug(f"Night evaluate error {symbol}: {exc}")

    final_candidates = sorted(
        final_candidates,
        key=lambda x: (x["Final_Night_Score"], x["Demand_Pressure"], x["Squeeze_Proxy"]),
        reverse=True,
    )
    return final_candidates, rejected
# ============================================================
# EKRANLAR
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "⚡ Canlı Gün İçi Radar",
        "🧠 Intraday Trade Engine",
        "🔮 Kurumsal Swing Radar",
        "📊 VWAP Analizi ve Emir Merkezi",
        "🌙 Night Buy Scanner",
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
# TAB 2 - INTRADAY TRADE ENGINE
# ============================================================
with tab2:
    st.subheader("🧠 Intraday Trade Engine")
    st.write("Aynı gün canlı verilerle trade edilebilecek güçlü setup'ları bulur. Mevcut continuation motorundan ayrıdır.")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        intraday_session = st.selectbox(
            "Tarama seansı",
            ["regular", "premarket", "afterhours"],
            index=0,
            format_func=lambda x: {"regular": "Regular", "premarket": "Pre-Market", "afterhours": "After-Hours"}[x],
        )
    with col_b:
        intraday_universe = st.slider("İlk tarama genişliği", min_value=10, max_value=60, value=25, step=5)
    with col_c:
        intraday_min_price = st.number_input("Min fiyat ($)", min_value=0.5, max_value=50.0, value=1.0, step=0.5, key='intraday_min_price')
    with col_d:
        intraday_max_price = st.number_input("Max fiyat ($)", min_value=5.0, max_value=500.0, value=100.0, step=5.0, key='intraday_max_price')

    st.caption("Open Drive Breakout, First Pullback ve VWAP Reclaim setup'larını tarar. MACD/RSI yardımcı, VWAP/EMA/hacim ana omurgadır.")

    if st.button("⚡ Intraday Engine'i Çalıştır", key="btn_intraday_engine"):
        try:
            with st.spinner("1. Aşama: İntraday evren taranıyor..."):
                universe_df = fetch_intraday_trade_universe(
                    session_name=intraday_session,
                    max_records=intraday_universe,
                    min_price=intraday_min_price,
                    max_price=intraday_max_price,
                )

            if universe_df.empty:
                st.warning("İntraday evren boş döndü.")
            else:
                tickers = universe_df['yahoo_symbol'].dropna().unique().tolist()
                if 'SPY' not in tickers:
                    tickers.append('SPY')

                with st.spinner("2. Aşama: Günlük referans veriler indiriliyor..."):
                    daily_dict = download_daily_data_chunked(
                        tickers,
                        period='260d',
                        chunk_size=10,
                        pause=1.0,
                        alpaca_key=api_key,
                        alpaca_secret=secret_key,
                        alpaca_feed=os.getenv('ALPACA_FEED', 'iex'),
                    )

                spy_daily_df = daily_dict.get('SPY', pd.DataFrame())

                with st.spinner("3. Aşama: Intraday setup scoring yapılıyor..."):
                    intraday_candidates, intraday_rejected = evaluate_intraday_candidates(
                        universe_df=universe_df,
                        daily_dict=daily_dict,
                        spy_daily_df=spy_daily_df,
                    )

                intraday_df = pd.DataFrame(intraday_candidates)
                if intraday_df.empty:
                    st.warning("Trade edilebilir intraday setup bulunamadı.")
                else:
                    st.success(f"Trade edilebilir intraday aday sayısı: {len(intraday_df)}")
                    st.dataframe(intraday_df, use_container_width=True)

                    top_intraday = intraday_df.head(3).copy()
                    st.subheader("🎯 En Güçlü Intraday 3")
                    st.dataframe(top_intraday, use_container_width=True)

                    csv_intraday = intraday_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        "📥 Intraday sonuçları CSV indir",
                        data=csv_intraday,
                        file_name=f"intraday_engine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv',
                    )

                with st.expander("İntraday reddedilenler"):
                    rej = pd.DataFrame(intraday_rejected)
                    if not rej.empty:
                        st.dataframe(rej, use_container_width=True)
                    else:
                        st.write("Kayıt yok.")

        except Exception as exc:
            st.error(f"Intraday engine hatası: {exc}")
            if DEBUG_MODE:
                st.code(traceback.format_exc())


# ============================================================
# TAB 2 - SWING RADAR
# ============================================================
with tab3:
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
        try:
            with st.spinner("1. Aşama: TradingView adayları taranıyor..."):
                tv_df = fetch_tradingview_candidates(algo_choice=algo_choice, max_records=max_scan_records)

            if tv_df.empty:
                st.warning("İlk aşamada aday bulunamadı.")
            else:
                st.info(f"İlk aşamada bulunan aday sayısı: {len(tv_df)}")

                yahoo_tickers = tv_df["yahoo_symbol"].dropna().unique().tolist()
                if "SPY" not in yahoo_tickers:
                    yahoo_tickers.append("SPY")

                with st.spinner("2. Aşama: Günlük veriler Alpaca/Yahoo üzerinden indiriliyor..."):
                    data_dict = download_daily_data_chunked(
                        yahoo_tickers,
                        period="220d",
                        chunk_size=10,
                        pause=2.0,
                        alpaca_key=api_key,
                        alpaca_secret=secret_key,
                        alpaca_feed=os.getenv("ALPACA_FEED", "iex"),
                    )

                with st.spinner("3. Aşama: İkinci filtre ve scoring uygulanıyor..."):
                    final_candidates, rejected_log = evaluate_candidates(algo_choice, tv_df, data_dict)

                gc.collect()

                final_df = pd.DataFrame(final_candidates)

                if final_df.empty:
                    st.warning("Kurallara uyan aday çıkmadı.")
                else:
                    st.success(f"Filtrelerden başarıyla geçen hisse sayısı: {len(final_df)}")

                    st.subheader("🏆 Top 10 Aday")
                    st.dataframe(final_df.head(10), use_container_width=True)

                    top3_df = rank_top3(final_df)

                    st.subheader("🥇 En İyi 3")
                    if not top3_df.empty:
                        st.dataframe(top3_df, use_container_width=True)
                    else:
                        st.warning("Final havuzundan entry-ready 3 hisse çıkmadı.")

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
                    rej_df = pd.DataFrame(rejected_log)
                    if not rej_df.empty:
                        st.dataframe(rej_df, use_container_width=True)
                    else:
                        st.write("Kayıt yok.")

        except Exception as exc:
            st.error(f"Tarama hatası: {exc}")
            if DEBUG_MODE:
                st.code(traceback.format_exc())


# ============================================================
# TAB 4 - ÇOKLU SEANS VWAP + EMİR MERKEZİ
# ============================================================
with tab4:
    st.subheader("📊 VWAP Analizi ve Emir Merkezi")

    ticker_input = st.text_input("İşlem yapılacak hisse sembolü", "").upper().strip()
    account_size = st.number_input("Hesap büyüklüğü ($)", min_value=100.0, value=2000.0, step=100.0)
    risk_pct = st.number_input("Trade başına risk (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.1) / 100

    if ticker_input:
        if st.button(f"🔍 {ticker_input} için çoklu seans VWAP analizi yap"):
            with st.spinner("Veriler çekiliyor ve seans bazlı VWAP hesaplanıyor..."):
                try:
                    yahoo_symbol = normalize_symbol_for_yahoo(ticker_input)

                    intraday = get_intraday_session_data(
                        symbol=ticker_input,
                        yahoo_symbol=yahoo_symbol,
                        api_key_value=api_key,
                        secret_key_value=secret_key,
                        alpaca_feed=os.getenv("ALPACA_FEED", "iex"),
                    )
                    daily_dict = download_daily_data_chunked([yahoo_symbol], period="260d", chunk_size=1, pause=0,
                                                           alpaca_key=api_key, alpaca_secret=secret_key,
                                                           alpaca_feed=os.getenv("ALPACA_FEED", "iex"))
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




# ============================================================
# TAB 5 - NIGHT BUY / OVERNIGHT PRESSURE ENGINE
# ============================================================
with tab5:
    st.subheader("🌙 Night Buy Scanner — Overnight Pressure Engine")
    st.write(
        "Bu modül gece/after-hours alım → ertesi gün premarket veya normal seansta satış stratejisi için "
        "talep basıncı, EMA/MACD/RSI uyumu, squeeze proxy, AH tutunma, Fibonacci/kanal alanı ve fakeout riskini birlikte puanlar."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        night_scan_mode = st.selectbox(
            "Tarama modu",
            ["afterhours", "premarket", "regular"],
            index=0,
            format_func=lambda x: {
                "afterhours": "After-Hours / Gece",
                "premarket": "Pre-Market",
                "regular": "Regular Momentum",
            }[x],
        )
    with c2:
        night_universe_size = st.slider("Evren genişliği", min_value=20, max_value=150, value=80, step=10, key="night_universe_size")
    with c3:
        night_min_price = st.number_input("Min fiyat ($)", min_value=0.5, max_value=20.0, value=1.0, step=0.5, key="night_min_price")
    with c4:
        night_max_price = st.number_input("Max fiyat ($)", min_value=3.0, max_value=500.0, value=80.0, step=5.0, key="night_max_price")

    c5, c6, c7 = st.columns(3)
    with c5:
        night_min_volume = st.number_input("Min regular hacim", min_value=50_000, max_value=5_000_000, value=150_000, step=50_000, key="night_min_volume")
    with c6:
        min_night_score = st.slider("Min Night Score", min_value=0, max_value=100, value=70, step=5, key="min_night_score")
    with c7:
        max_fakeout = st.slider("Max Fakeout Risk", min_value=0, max_value=100, value=60, step=5, key="max_fakeout")

    st.caption(
        "Not: Bu ilk sürümde squeeze, gerçek short-float verisi olmadan düşük piyasa değeri + RVOL + AH hacim + kırılım davranışıyla proxy olarak hesaplanır. "
        "Adaylar gerçek para için değil, önce paper trading ve sonuç kaydı için kullanılmalıdır."
    )

    if st.button("🌙 Night Buy Scanner'ı Çalıştır", key="btn_night_buy"):
        try:
            with st.spinner("1. Aşama: Night-buy evreni TradingView üzerinden taranıyor..."):
                night_universe = fetch_night_buy_universe(
                    scan_mode=night_scan_mode,
                    max_records=night_universe_size,
                    min_price=night_min_price,
                    max_price=night_max_price,
                    min_volume=int(night_min_volume),
                )

            if night_universe.empty:
                st.warning("Night-buy evreni boş döndü. Seans kapalıysa veya TradingView postmarket verisi gelmiyorsa premarket/regular modunu dene.")
            else:
                st.info(f"İlk aşamada bulunan aday sayısı: {len(night_universe)}")
                tickers = night_universe["yahoo_symbol"].dropna().unique().tolist()

                with st.spinner("2. Aşama: Günlük veri indiriliyor..."):
                    night_daily_dict = download_daily_data_chunked(
                        tickers,
                        period="260d",
                        chunk_size=10,
                        pause=1.0,
                        alpaca_key=api_key,
                        alpaca_secret=secret_key,
                        alpaca_feed=os.getenv("ALPACA_FEED", "iex"),
                    )

                with st.spinner("3. Aşama: Night Pressure / Squeeze / Fakeout scoring yapılıyor..."):
                    night_candidates, night_rejected = evaluate_night_buy_candidates(night_universe, night_daily_dict)

                night_all_df = pd.DataFrame(night_candidates)
                show_cols = [
                    "Symbol", "Grade", "Status", "Final_Night_Score", "Demand_Pressure", "Technical_Alignment",
                    "Squeeze_Proxy", "AH_Strength", "Channel_Clarity", "Fakeout_Risk",
                    "AH_Live_Price", "AH_Change_%", "RVOL20", "AH_Vol_to_Daily_%",
                    "EMA9_gt_EMA20", "RSI14", "MACD_Rising",
                    "Night_Entry_Low", "Night_Entry_High", "Stop", "TP1", "TP2", "Why", "Risk_Notes", "Aggressive_Reason"
                ]

                if night_all_df.empty:
                    trade_df = pd.DataFrame()
                    watch_df = pd.DataFrame()
                    aggressive_df = pd.DataFrame()
                    rejected_scored_df = pd.DataFrame()
                    st.warning("Night Buy scoring üretilemedi. Veri/bağlantı reddedilenler bölümünü kontrol et.")
                else:
                    night_all_df = night_all_df.sort_values(
                        ["Final_Night_Score", "Demand_Pressure", "Squeeze_Proxy"],
                        ascending=False,
                    ).copy()

                    trade_df = night_all_df[
                        (night_all_df["Status"] == "TRADE_CANDIDATE") &
                        (night_all_df["Final_Night_Score"] >= max(75, min_night_score)) &
                        (night_all_df["Fakeout_Risk"] <= max_fakeout) &
                        (night_all_df["Hard_Reject"] == False)
                    ].copy()

                    watch_df = night_all_df[
                        (night_all_df["Status"] == "PAPER_WATCHLIST") &
                        (night_all_df["Final_Night_Score"] >= min_night_score) &
                        (night_all_df["Fakeout_Risk"] <= max_fakeout) &
                        (night_all_df["Hard_Reject"] == False)
                    ].copy()

                    aggressive_df = night_all_df[
                        (night_all_df["Status"] == "AGGRESSIVE_SQUEEZE_WATCH") &
                        (night_all_df["Fakeout_Risk"] <= max_fakeout) &
                        (night_all_df["Hard_Reject"] == False)
                    ].copy()

                    shown_symbols = pd.concat(
                        [trade_df["Symbol"], watch_df["Symbol"], aggressive_df["Symbol"]],
                        ignore_index=True
                    )
                    rejected_scored_df = night_all_df[
                        ~night_all_df["Symbol"].isin(shown_symbols)
                    ].copy()

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("A/B Night Buy", len(trade_df))
                    m2.metric("Paper Watchlist", len(watch_df))
                    m3.metric("Aggressive Squeeze", len(aggressive_df))
                    m4.metric("Reject / zayıf", len(rejected_scored_df) + len(night_rejected))

                    available_cols = [c for c in show_cols if c in night_all_df.columns]

                    if trade_df.empty:
                        st.warning("Gerçek Night Buy adayı yok. Bu gece için sistem işlem baskısı yeterli görmedi.")
                    else:
                        st.success(f"A/B kalite Night Buy adayı: {len(trade_df)}")
                        st.subheader("🏆 A/B Night Buy Adayları")
                        st.dataframe(trade_df[available_cols].head(20), use_container_width=True)

                        st.subheader("🥇 En Güçlü 3 Night Buy")
                        st.dataframe(trade_df[available_cols].head(3), use_container_width=True)

                    if watch_df.empty:
                        st.info("Paper Watchlist adayı yok. Min Night Score değerini 55-60 bandına çekerek eğitim amaçlı izleme yapılabilir.")
                    else:
                        st.subheader("📝 Paper Watchlist — işlem değil, takip/eğitim")
                        st.caption("Bu tablo 60-74 arası adayları gösterir. Gerçek işlem için değil; ertesi gün sonuç kaydı ve ağırlık kalibrasyonu içindir.")
                        st.dataframe(watch_df[available_cols].head(25), use_container_width=True)

                    if aggressive_df.empty:
                        st.info("Aggressive Squeeze Watch adayı yok. Bu iyi bir şey olabilir; sistem riskli squeeze ihtimalini zorlamıyor.")
                    else:
                        st.subheader("🔥 Aggressive Squeeze Watch — yüksek riskli, gerçek işlem değil")
                        st.caption(
                            "Bu tablo final skoru 60 altı kalsa bile squeeze/talep/AH momentum basıncı yüksek olan adayları gösterir. "
                            "Gece alım kararı değil; premarket/canlı teyit ve paper takip listesidir."
                        )
                        st.dataframe(aggressive_df[available_cols].head(25), use_container_width=True)

                    csv_all = night_all_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 Tüm Night Buy skorlarını CSV indir",
                        data=csv_all,
                        file_name=f"night_buy_all_scored_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

                    if not trade_df.empty:
                        csv_trade = trade_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "📥 Sadece A/B adayları CSV indir",
                            data=csv_trade,
                            file_name=f"night_buy_trade_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )

                    if not watch_df.empty:
                        csv_watch = watch_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "📥 Paper Watchlist CSV indir",
                            data=csv_watch,
                            file_name=f"night_buy_paper_watchlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )

                    if not aggressive_df.empty:
                        csv_aggressive = aggressive_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "📥 Aggressive Squeeze Watch CSV indir",
                            data=csv_aggressive,
                            file_name=f"night_buy_aggressive_squeeze_watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )

                with st.expander("Night Buy reject / veri hatası / zayıf skor detayları"):
                    if 'rejected_scored_df' in locals() and not rejected_scored_df.empty:
                        reject_cols = [c for c in show_cols if c in rejected_scored_df.columns]
                        st.write("Skorlandı ama işlem/watchlist filtresine girmedi:")
                        st.dataframe(rejected_scored_df[reject_cols].head(100), use_container_width=True)

                    rej_df = pd.DataFrame(night_rejected)
                    if not rej_df.empty:
                        st.write("Veri eksikliği veya hesaplama nedeniyle skorlanamayanlar:")
                        st.dataframe(rej_df, use_container_width=True)
                    elif (('rejected_scored_df' not in locals()) or rejected_scored_df.empty):
                        st.write("Kayıt yok.")

        except Exception as exc:
            st.error(f"Night Buy Scanner hatası: {exc}")
            if DEBUG_MODE:
                st.code(traceback.format_exc())
st.write("---")
st.caption(
    "Bu araç yatırım tavsiyesi değildir. Önce paper trading ile test edilmesi gerekir. "
    "Dış veri kaynaklarında rate-limit ve veri uyuşmazlığı olabilir."
)
