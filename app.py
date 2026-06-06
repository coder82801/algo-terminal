
import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(page_title="Algo Terminal Overnight + Ignition", layout="wide")
st.title("🎯 Algo Terminal — Overnight + Ignition")
st.caption(
    "Nihai sürüm: Overnight Engine + Premarket Ignition + Continuation + Supernova + ML + Risk Monitor. "
    "Kural tabanlı motorlar korunur; varsa eğitilmiş ML modelleri olasılık katmanı olarak eklenir."
)

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

DEFAULT_CONTINUATION_SYMBOLS = [
    "ROLR", "OGN", "RBBN", "LWLG", "AAOI", "TTMI", "ALAB",
    "CUE", "RAYA", "SQFT", "SIDU", "FUSE", "TPST", "IPST"
]

DEFAULT_SUPERNOVA_SYMBOLS = [
    "RMSG", "STI", "TNON", "SOPA", "CTMX", "RR", "SIDU",
    "MAXN", "CREG", "SKYQ", "SQFT", "FUSE", "GN", "CUE"
]

DEFAULT_ML_TRAIN_SYMBOLS = sorted(list(dict.fromkeys(
    DEFAULT_CONTINUATION_SYMBOLS + DEFAULT_SUPERNOVA_SYMBOLS +
    ["EOSE", "CTMX", "SOPA", "RR", "TNON", "IREN", "RMSG", "STI", "MAXN"]
)))

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_FEED = (os.getenv("ALPACA_FEED", "iex") or "iex").lower()
ALPACA_TRADING_BASE = os.getenv("ALPACA_TRADING_BASE", "https://paper-api.alpaca.markets")

MODEL_DIR = os.getenv("MODEL_DIR", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MIN_PRICE = 0.10
MAX_PRICE_CONTINUATION = 50.0
MAX_PRICE_SUPERNOVA = 5.0

DEFAULT_CONT_SCORE_AL = 62
DEFAULT_CONT_SCORE_STRONG = 78
DEFAULT_SUPER_PATTERN_AL = 60
DEFAULT_SUPER_PATTERN_STRONG = 72
DEFAULT_MIN_PRE_DOLLAR_VOL = 75_000
DEFAULT_MAX_EXTENSION = 0.08

SYMBOL_FLAGS = {
    "RMSG": {"recent_reverse_split": False, "recent_deficiency": True, "otc_risk": False, "catalyst_fresh": True},
    "STI": {"recent_reverse_split": False, "recent_deficiency": False, "otc_risk": False, "catalyst_fresh": True},
    "SOPA": {"recent_reverse_split": False, "recent_deficiency": False, "otc_risk": False, "catalyst_fresh": False},
    "TNON": {"recent_reverse_split": False, "recent_deficiency": False, "otc_risk": False, "catalyst_fresh": False},
    "IREN": {"recent_reverse_split": False, "recent_deficiency": False, "otc_risk": False, "catalyst_fresh": True},
}

CONT_FEATURE_COLS = [
    "price",
    "prev_day_ret",
    "prev_close_strength",
    "prev_vol_ratio",
    "prev_dollar_vol",
    "ema9_dist_pct",
    "ema20_dist_pct",
    "atr_pct",
    "range_pct_1d",
    "range_pct_10d_avg",
    "base_tightness10",
    "drawdown90",
    "rebound30",
]

SUPER_FEATURE_COLS = [
    "price",
    "prev_day_ret",
    "prev_close_strength",
    "prev_vol_ratio",
    "prev_dollar_vol",
    "range_expansion",
    "base_tightness10",
    "drawdown90",
    "rebound30",
    "breakout20",
    "ema9_dist_pct",
    "ema20_dist_pct",
]


# ============================================================
# HELPERS
# ============================================================
def safe_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def clamp(x, low, high):
    return max(low, min(high, x))


def avg(arr):
    vals = [safe_float(x, np.nan) for x in arr]
    vals = [x for x in vals if pd.notna(x)]
    return float(np.mean(vals)) if vals else np.nan


def median(arr):
    vals = [safe_float(x, np.nan) for x in arr]
    vals = [x for x in vals if pd.notna(x)]
    return float(np.median(vals)) if vals else np.nan


def round_smart(v):
    if pd.isna(v):
        return np.nan
    v = float(v)
    if abs(v) < 1:
        return round(v, 4)
    if abs(v) < 10:
        return round(v, 3)
    return round(v, 2)


def format_price(v):
    if pd.isna(v):
        return "-"
    v = float(v)
    if v < 1:
        return f"{v:.4f}"
    if v < 10:
        return f"{v:.3f}"
    return f"{v:.2f}"


def parse_symbols(text):
    parts = [x.strip().upper() for x in str(text or "").split(",")]
    return list(dict.fromkeys([x for x in parts if x]))


def get_flags(symbol):
    return {
        "recent_reverse_split": False,
        "recent_deficiency": False,
        "otc_risk": False,
        "catalyst_fresh": False,
        **SYMBOL_FLAGS.get(str(symbol).upper(), {})
    }


def now_ny():
    return datetime.now(NY_TZ)


def ny_date_str(dt=None):
    dt = dt or now_ny()
    return dt.astimezone(NY_TZ).strftime("%Y-%m-%d")


def ny_time_str(dt=None):
    dt = dt or now_ny()
    return dt.astimezone(NY_TZ).strftime("%H:%M:%S")


def get_session_label(dt=None):
    dt = dt or now_ny()
    t = dt.astimezone(NY_TZ).time()
    wd = dt.astimezone(NY_TZ).weekday()

    if wd >= 5:
        return "weekend"
    if t >= datetime.strptime("04:00", "%H:%M").time() and t < datetime.strptime("09:30", "%H:%M").time():
        return "premarket"
    if t >= datetime.strptime("09:30", "%H:%M").time() and t < datetime.strptime("16:00", "%H:%M").time():
        return "regular"
    if t >= datetime.strptime("16:00", "%H:%M").time() and t <= datetime.strptime("20:00", "%H:%M").time():
        return "afterhours"
    return "closed"


def local_date_from_iso(iso_str, tz=NY_TZ):
    dt = pd.Timestamp(iso_str)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    return dt.tz_convert(tz).strftime("%Y-%m-%d")


def local_time_from_iso(iso_str, tz=NY_TZ):
    dt = pd.Timestamp(iso_str)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    return dt.tz_convert(tz).strftime("%H:%M:%S")


def decision_sort_key(decision):
    return {"GÜÇLÜ AL": 0, "AL": 1, "İZLE": 2, "ALMA": 3}.get(decision, 9)


def top_cards(df, n=3):
    return [] if df is None or df.empty else df.head(n).to_dict("records")


def apply_session_overlay(df, session):
    if df is None or df.empty:
        return df

    out = df.copy()

    if session in ["afterhours", "closed", "weekend"]:
        out["Trade_Phase"] = "NIGHTLY_WATCHLIST"
        out["Action_Status"] = "PREMARKET_REQUIRED"
        if "Decision" in out.columns:
            out["Decision"] = out["Decision"].replace({"GÜÇLÜ AL": "İZLE", "AL": "İZLE"})
        if "Real_Money_Allowed" in out.columns:
            out["Real_Money_Allowed"] = False
        if "Risk_Grade" in out.columns:
            out["Risk_Grade"] = "WATCHLIST_ONLY"
        if "Risk_Tags" in out.columns:
            out["Risk_Tags"] = out["Risk_Tags"].fillna("").apply(
                lambda x: (x + " | session_closed_watchlist").strip(" |")
            )
        return out

    if session == "premarket":
        out["Trade_Phase"] = "PREMARKET_ACTIONABLE"
        out["Action_Status"] = "LIVE_PREMARKET"
        return out

    if session == "regular":
        out["Trade_Phase"] = "REGULAR_REVIEW"
        out["Action_Status"] = "OPEN_MARKET"
        return out

    out["Trade_Phase"] = "BACKTEST"
    out["Action_Status"] = "BACKTEST"
    return out


def render_session_message(session):
    if session in ["afterhours", "closed", "weekend"]:
        st.info("Piyasa kapalı. Bu çıktı işlem sinyali değil, Nightly Watchlist olarak okunmalı. Gerçek işlem için premarket teyidi gerekir.")
    elif session == "premarket":
        st.success("Premarket açık. Bu oturumda çıkan AL / GÜÇLÜ AL sinyalleri actionable kabul edilebilir.")
    elif session == "regular":
        st.warning("Market open modundasın. Bu çıktı açılış sonrası gözden geçirme içindir; geç kalma ve spread riskini ayrıca izle.")
    elif session == "backtest":
        st.caption("Backtest modunda geçmiş gün simülasyonu gösteriliyor.")


def nz(v):
    return 0.0 if pd.isna(v) else float(v)


# ============================================================
# DATA LAYER
# ============================================================
def have_alpaca():
    return bool(ALPACA_API_KEY and ALPACA_SECRET_KEY)


def alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }


def fetch_alpaca_bars(symbols, timeframe="1Day", start=None, end=None, feed=None, limit=10000):
    if not have_alpaca():
        return {}

    if isinstance(symbols, str):
        symbols = [symbols]
    if not symbols:
        return {}

    url = "https://data.alpaca.markets/v2/stocks/bars"
    params = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "adjustment": "raw",
        "sort": "asc",
        "limit": str(limit),
        "feed": feed or ALPACA_FEED,
    }

    out = {s: [] for s in symbols}
    page_token = None

    while True:
        req_params = params.copy()
        if page_token:
            req_params["page_token"] = page_token

        try:
            r = requests.get(url, headers=alpaca_headers(), params=req_params, timeout=30)
        except Exception:
            return out

        if r.status_code != 200:
            return out

        payload = r.json()
        bars_map = payload.get("bars", {})
        for sym, bars in bars_map.items():
            out.setdefault(sym, [])
            out[sym].extend(bars)

        page_token = payload.get("next_page_token")
        if not page_token:
            break

    for sym in out:
        out[sym] = sorted(out[sym], key=lambda x: x.get("t", ""))

    return out


def fetch_alpaca_latest_quotes(symbols):
    if not have_alpaca():
        return {}

    if isinstance(symbols, str):
        symbols = [symbols]
    if not symbols:
        return {}

    url = "https://data.alpaca.markets/v2/stocks/quotes/latest"
    params = {"symbols": ",".join(symbols), "feed": ALPACA_FEED}
    try:
        r = requests.get(url, headers=alpaca_headers(), params=params, timeout=20)
        if r.status_code != 200:
            return {}
        payload = r.json()
        return payload.get("quotes", {})
    except Exception:
        return {}


def daily_bars_to_df(bars):
    if not bars:
        return pd.DataFrame()

    rows = []
    for b in bars:
        rows.append({
            "Date": pd.to_datetime(b["t"], utc=True),
            "Open": safe_float(b.get("o")),
            "High": safe_float(b.get("h")),
            "Low": safe_float(b.get("l")),
            "Close": safe_float(b.get("c")),
            "Volume": safe_float(b.get("v")),
        })

    df = pd.DataFrame(rows).dropna()
    if df.empty:
        return df
    return df.set_index("Date").sort_index()


def minute_bars_to_df(bars):
    if not bars:
        return pd.DataFrame()

    rows = []
    for b in bars:
        rows.append({
            "Date": pd.to_datetime(b["t"], utc=True),
            "Open": safe_float(b.get("o")),
            "High": safe_float(b.get("h")),
            "Low": safe_float(b.get("l")),
            "Close": safe_float(b.get("c")),
            "Volume": safe_float(b.get("v")),
        })

    df = pd.DataFrame(rows).dropna()
    if df.empty:
        return df
    return df.set_index("Date").sort_index()


def fetch_yahoo_daily(symbol, lookback_days=420):
    try:
        period = "2y" if lookback_days > 400 else "1y"
        df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


def get_daily_df(symbol, lookback_days=420):
    if have_alpaca():
        end_dt = datetime.now(UTC_TZ) + timedelta(days=1)
        start_dt = end_dt - timedelta(days=lookback_days)
        bars_map = fetch_alpaca_bars(
            [symbol],
            timeframe="1Day",
            start=start_dt.isoformat().replace("+00:00", "Z"),
            end=end_dt.isoformat().replace("+00:00", "Z"),
            feed=ALPACA_FEED,
            limit=10000,
        )
        df = daily_bars_to_df(bars_map.get(symbol, []))
        if not df.empty:
            return df
    return fetch_yahoo_daily(symbol, lookback_days=lookback_days)


def fetch_minute_df(symbol, start_dt_utc, end_dt_utc, timeframe="1Min"):
    if not have_alpaca():
        return pd.DataFrame()
    bars_map = fetch_alpaca_bars(
        [symbol],
        timeframe=timeframe,
        start=start_dt_utc.isoformat().replace("+00:00", "Z"),
        end=end_dt_utc.isoformat().replace("+00:00", "Z"),
        feed=ALPACA_FEED,
        limit=10000,
    )
    return minute_bars_to_df(bars_map.get(symbol, []))


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_us_equity_universe(exchanges=("NASDAQ", "NYSE")):
    if not have_alpaca():
        return []

    url = f"{ALPACA_TRADING_BASE}/v2/assets"
    params = {"status": "active", "asset_class": "us_equity"}
    exchange_set = {str(x).upper() for x in exchanges}

    try:
        r = requests.get(url, headers=alpaca_headers(), params=params, timeout=40)
        if r.status_code != 200:
            return []
        payload = r.json()
    except Exception:
        return []

    universe = []
    for asset in payload:
        sym = str(asset.get("symbol", "")).upper().strip()
        exch = str(asset.get("exchange", "")).upper().strip()
        tradable = bool(asset.get("tradable", False))

        if not sym or not tradable:
            continue
        if exch not in exchange_set:
            continue
        if "/" in sym or " " in sym:
            continue

        universe.append(sym)

    return sorted(list(dict.fromkeys(universe)))


def build_fast_daily_feature_row(symbol, daily_df, trade_date_str):
    ctx = get_trade_date_context(daily_df, trade_date_str)
    if ctx is None:
        return None

    last = ctx["last"]
    prev = ctx["prev"]
    hist10 = ctx["hist10"]
    hist20 = ctx["hist20"]
    hist60 = ctx["hist60"]
    hist90 = ctx["hist90"]

    price = safe_float(last["Close"], np.nan)
    prev_close = safe_float(prev["Close"], np.nan)
    prev_day_ret = ((price - prev_close) / prev_close * 100.0) if prev_close > 0 else np.nan
    prev_close_strength = closing_strength(price, safe_float(last["Low"], np.nan), safe_float(last["High"], np.nan))
    prev_close_strength_pct = prev_close_strength * 100.0 if pd.notna(prev_close_strength) else np.nan

    avg_vol20 = max(avg(hist20["Volume"].iloc[:-1] if len(hist20) > 1 else hist20["Volume"]), 1)
    prev_vol_ratio = safe_float(last["Volume"], 0) / avg_vol20
    prev_dollar_vol = price * safe_float(last["Volume"], 0)

    ema9 = calc_ema(hist20["Close"], 9).iloc[-1] if len(hist20) >= 9 else np.nan
    ema20 = calc_ema(hist20["Close"], 20).iloc[-1] if len(hist20) >= 20 else np.nan

    atr14 = calc_atr(hist20, 14).iloc[-1] if len(hist20) >= 14 else np.nan
    atr_pct = (atr14 / price * 100.0) if pd.notna(atr14) and price > 0 else np.nan

    range_pct_1d = range_pct(safe_float(last["High"], np.nan), safe_float(last["Low"], np.nan), price)
    range_pct_10d_avg = avg([range_pct(h, l, c) for h, l, c in zip(hist10["High"], hist10["Low"], hist10["Close"])])

    hist10_high = safe_float(hist10["High"].max(), np.nan)
    hist10_low = safe_float(hist10["Low"].min(), np.nan)
    high20_ex_last = safe_float(hist20["High"].iloc[:-1].max() if len(hist20) > 1 else hist20["High"].max(), np.nan)
    low30 = safe_float(hist60["Low"].iloc[-30:].min() if len(hist60) >= 30 else hist60["Low"].min(), np.nan)
    high90 = safe_float(hist90["High"].max(), np.nan)

    drawdown90 = ((price / high90) - 1) * 100.0 if pd.notna(high90) and high90 > 0 else np.nan
    rebound30 = ((price - low30) / low30) * 100.0 if pd.notna(low30) and low30 > 0 else np.nan
    base_tightness10 = ((hist10_high - hist10_low) / price * 100.0) if pd.notna(price) and price > 0 else np.nan
    breakout20 = bool(pd.notna(high20_ex_last) and price > high20_ex_last)

    ema9_dist_pct = ((price - ema9) / ema9 * 100.0) if pd.notna(ema9) and ema9 > 0 else np.nan
    ema20_dist_pct = ((price - ema20) / ema20 * 100.0) if pd.notna(ema20) and ema20 > 0 else np.nan

    return {
        "Symbol": symbol,
        "price": price,
        "prev_day_ret": prev_day_ret,
        "prev_close_strength": prev_close_strength_pct,
        "prev_vol_ratio": prev_vol_ratio,
        "prev_dollar_vol": prev_dollar_vol,
        "ema9": ema9,
        "ema20": ema20,
        "ema9_dist_pct": ema9_dist_pct,
        "ema20_dist_pct": ema20_dist_pct,
        "atr_pct": atr_pct,
        "range_pct_1d": range_pct_1d,
        "range_pct_10d_avg": range_pct_10d_avg,
        "drawdown90": drawdown90,
        "rebound30": rebound30,
        "base_tightness10": base_tightness10,
        "breakout20": breakout20,
    }


def fast_prefilter_score(engine_name, feat):
    if engine_name == "overnight":
        engine_name = "continuation"
    elif engine_name == "premarket_ignition":
        engine_name = "supernova"
    price = safe_float(feat.get("price"), np.nan)
    prev_day_ret = safe_float(feat.get("prev_day_ret"), np.nan)
    prev_close_strength = safe_float(feat.get("prev_close_strength"), np.nan)
    prev_vol_ratio = safe_float(feat.get("prev_vol_ratio"), np.nan)
    prev_dollar_vol = safe_float(feat.get("prev_dollar_vol"), np.nan)
    drawdown90 = safe_float(feat.get("drawdown90"), np.nan)
    rebound30 = safe_float(feat.get("rebound30"), np.nan)
    base_tightness10 = safe_float(feat.get("base_tightness10"), np.nan)
    breakout20 = bool(feat.get("breakout20", False))
    ema9 = safe_float(feat.get("ema9"), np.nan)
    ema20 = safe_float(feat.get("ema20"), np.nan)

    score = 0.0

    if engine_name == "continuation":
        if pd.isna(price) or price < MIN_PRICE or price > MAX_PRICE_CONTINUATION:
            return -9999

        if 2 <= prev_day_ret < 12:
            score += 18
        elif 12 <= prev_day_ret < 35:
            score += 26
        elif 35 <= prev_day_ret < 70:
            score += 12
        elif prev_day_ret < 0:
            score -= 10

        if prev_close_strength >= 80:
            score += 18
        elif prev_close_strength >= 65:
            score += 10
        elif prev_close_strength < 45:
            score -= 8

        if prev_vol_ratio >= 1.5:
            score += 12
        if prev_vol_ratio >= 3:
            score += 8

        if prev_dollar_vol >= 500_000:
            score += 14
        elif prev_dollar_vol < 100_000:
            score -= 10

        if pd.notna(ema9) and pd.notna(ema20) and price > ema9 > ema20:
            score += 12

        if pd.notna(base_tightness10) and 3 <= base_tightness10 <= 30:
            score += 4

        return score

    if pd.isna(price) or price < MIN_PRICE or price > MAX_PRICE_SUPERNOVA:
        return -9999

    if -90 <= drawdown90 <= -20:
        score += 18
    elif -20 < drawdown90 <= -5:
        score += 6

    if 4 <= base_tightness10 <= 35:
        score += 14
    elif base_tightness10 > 70:
        score -= 8

    if 3 <= prev_day_ret < 20:
        score += 12
    elif 20 <= prev_day_ret < 60:
        score += 18
    elif 60 <= prev_day_ret < 120:
        score += 10

    if prev_close_strength >= 80:
        score += 16
    elif prev_close_strength >= 65:
        score += 8

    if prev_vol_ratio >= 2:
        score += 14
    if prev_vol_ratio >= 5:
        score += 10

    if prev_dollar_vol >= 200_000:
        score += 12
    elif prev_dollar_vol < 50_000:
        score -= 10

    if 0 <= rebound30 <= 150:
        score += 6

    if breakout20:
        score += 8

    return score


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def build_full_market_prefilter(engine_name, trade_date_str, exchanges=("NASDAQ", "NYSE"), prefilter_limit=120, universe_limit=1500, chunk_size=250):
    universe = fetch_us_equity_universe(exchanges)
    if universe_limit and universe_limit > 0:
        universe = universe[:universe_limit]

    if not universe:
        return pd.DataFrame(), 0, 0

    trade_date = datetime.strptime(trade_date_str, "%Y-%m-%d").replace(tzinfo=NY_TZ)
    end_dt = trade_date.astimezone(UTC_TZ) + timedelta(days=1)
    start_dt = end_dt - timedelta(days=140)

    results = []
    for chunk in chunked(universe, chunk_size):
        bars_map = fetch_alpaca_bars(
            chunk,
            timeframe="1Day",
            start=start_dt.isoformat().replace("+00:00", "Z"),
            end=end_dt.isoformat().replace("+00:00", "Z"),
            feed=ALPACA_FEED,
            limit=10000
        )

        for sym in chunk:
            daily_df = daily_bars_to_df(bars_map.get(sym, []))
            if daily_df.empty:
                continue

            feat = build_fast_daily_feature_row(sym, daily_df, trade_date_str)
            if feat is None:
                continue

            feat["Fast_Score"] = fast_prefilter_score(engine_name, feat)
            if feat["Fast_Score"] > -999:
                results.append(feat)

    if not results:
        return pd.DataFrame(), len(universe), 0

    out = pd.DataFrame(results).sort_values("Fast_Score", ascending=False).reset_index(drop=True)
    total_ranked = len(out)
    out = out.head(prefilter_limit).copy()
    return out, len(universe), total_ranked


def resolve_scan_symbols(manual_text, use_full_market, engine_name, trade_date_str, exchanges, prefilter_limit, deep_scan_limit, universe_limit=0):
    if not use_full_market:
        syms = parse_symbols(manual_text)
        return syms, {"source": "manual", "universe_count": len(syms), "ranked_count": len(syms), "deep_count": len(syms)}, pd.DataFrame()

    pre_df, universe_count, ranked_count = build_full_market_prefilter(
        engine_name=engine_name,
        trade_date_str=trade_date_str,
        exchanges=tuple(exchanges),
        prefilter_limit=max(prefilter_limit, deep_scan_limit),
        universe_limit=universe_limit,
    )
    syms = pre_df["Symbol"].head(deep_scan_limit).tolist() if not pre_df.empty else []
    meta = {
        "source": "full_market",
        "universe_count": universe_count,
        "ranked_count": ranked_count,
        "deep_count": len(syms),
    }
    return syms, meta, pre_df


# ============================================================
# FEATURES / INDICATORS
# ============================================================
def calc_true_range(df):
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def calc_atr(df, period=14):
    return calc_true_range(df).rolling(period).mean()


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_vwap(df):
    if df is None or df.empty:
        return np.nan
    temp = df.dropna(subset=["High", "Low", "Close", "Volume"]).copy()
    if temp.empty:
        return np.nan
    tp = (temp["High"] + temp["Low"] + temp["Close"]) / 3
    vp = tp * temp["Volume"]
    vol = temp["Volume"].sum()
    return np.nan if vol <= 0 else float(vp.sum() / vol)


def closing_strength(close_, low_, high_):
    if pd.isna(close_) or pd.isna(low_) or pd.isna(high_):
        return np.nan
    r = high_ - low_
    if r <= 0:
        return 0.5
    return (close_ - low_) / r


def range_pct(high_, low_, ref):
    if pd.isna(high_) or pd.isna(low_) or pd.isna(ref) or ref <= 0:
        return np.nan
    return ((high_ - low_) / ref) * 100.0


def get_trade_date_context(daily_df, trade_date_str):
    if daily_df is None or daily_df.empty:
        return None

    temp = daily_df.copy()
    temp["NYDate"] = [local_date_from_iso(x) for x in temp.index]
    prior = temp[temp["NYDate"] < trade_date_str].copy()

    if len(prior) < 25:
        return None

    last = prior.iloc[-1]
    prev = prior.iloc[-2]
    hist10 = prior.iloc[-10:].copy()
    hist20 = prior.iloc[-20:].copy()
    hist60 = prior.iloc[-60:].copy() if len(prior) >= 60 else prior.copy()
    hist90 = prior.iloc[-90:].copy() if len(prior) >= 90 else prior.copy()

    return {
        "last": last,
        "prev": prev,
        "hist10": hist10,
        "hist20": hist20,
        "hist60": hist60,
        "hist90": hist90,
        "prior_dates": prior["NYDate"].drop_duplicates().tolist()[-10:]
    }


def get_premarket_context(minute_df, trade_date_str, cutoff_time="09:25:00"):
    if minute_df is None or minute_df.empty:
        return {
            "source": "NONE",
            "price": np.nan,
            "pre_high": np.nan,
            "pre_low": np.nan,
            "pre_vol": 0,
            "pre_vwap": np.nan,
            "hold_quality": np.nan,
            "pre_range_pct": np.nan,
            "bars": pd.DataFrame()
        }

    temp = minute_df.copy()
    temp["NYDate"] = [local_date_from_iso(x) for x in temp.index]
    temp["NYTime"] = [local_time_from_iso(x) for x in temp.index]

    same_day = temp[
        (temp["NYDate"] == trade_date_str) &
        (temp["NYTime"] >= "04:00:00") &
        (temp["NYTime"] <= cutoff_time)
    ].copy()

    if same_day.empty:
        return {
            "source": "NONE",
            "price": np.nan,
            "pre_high": np.nan,
            "pre_low": np.nan,
            "pre_vol": 0,
            "pre_vwap": np.nan,
            "hold_quality": np.nan,
            "pre_range_pct": np.nan,
            "bars": pd.DataFrame()
        }

    price = safe_float(same_day["Close"].iloc[-1], np.nan)
    ph = safe_float(same_day["High"].max(), np.nan)
    pl = safe_float(same_day["Low"].min(), np.nan)
    pv = safe_float(same_day["Volume"].sum(), 0)
    vwap = calc_vwap(same_day)

    hold = np.nan
    if pd.notna(ph) and pd.notna(pl) and ph > pl and pd.notna(price):
        hold = ((price - pl) / (ph - pl)) * 100.0

    return {
        "source": "REAL_PREMARKET",
        "price": price,
        "pre_high": ph,
        "pre_low": pl,
        "pre_vol": pv,
        "pre_vwap": vwap,
        "hold_quality": hold,
        "pre_range_pct": range_pct(ph, pl, price),
        "bars": same_day
    }


def premarket_baseline_ratio(minute_df, prior_dates, trade_date_str, cutoff_time="09:25:00"):
    if minute_df is None or minute_df.empty or not prior_dates:
        return np.nan, np.nan, 0

    temp = minute_df.copy()
    temp["NYDate"] = [local_date_from_iso(x) for x in temp.index]
    temp["NYTime"] = [local_time_from_iso(x) for x in temp.index]

    vols = []
    for d in prior_dates:
        if d == trade_date_str:
            continue
        day_df = temp[
            (temp["NYDate"] == d) &
            (temp["NYTime"] >= "04:00:00") &
            (temp["NYTime"] <= cutoff_time)
        ]
        v = safe_float(day_df["Volume"].sum() if not day_df.empty else 0, 0)
        if v > 0:
            vols.append(v)

    if not vols:
        return np.nan, np.nan, 0
    return median(vols), avg(vols), len(vols)


def extract_snapshot_features_from_ctx(ctx):
    last = ctx["last"]
    prev = ctx["prev"]
    hist10 = ctx["hist10"]
    hist20 = ctx["hist20"]
    hist60 = ctx["hist60"]
    hist90 = ctx["hist90"]

    price = safe_float(last["Close"], np.nan)
    prev_close = safe_float(prev["Close"], np.nan)
    prev_day_ret = ((price - prev_close) / prev_close * 100.0) if prev_close > 0 else np.nan
    prev_close_strength = closing_strength(price, safe_float(last["Low"], np.nan), safe_float(last["High"], np.nan))

    avg_vol20 = max(avg(hist20["Volume"].iloc[:-1] if len(hist20) > 1 else hist20["Volume"]), 1)
    prev_vol_ratio = safe_float(last["Volume"], 0) / avg_vol20
    prev_dollar_vol = price * safe_float(last["Volume"], 0)

    ema9 = calc_ema(hist20["Close"], 9).iloc[-1] if len(hist20) >= 9 else np.nan
    ema20 = calc_ema(hist20["Close"], 20).iloc[-1] if len(hist20) >= 20 else np.nan
    atr14 = calc_atr(hist20, 14).iloc[-1] if len(hist20) >= 14 else np.nan

    ema9_dist_pct = ((price - ema9) / ema9 * 100.0) if pd.notna(ema9) and ema9 > 0 else np.nan
    ema20_dist_pct = ((price - ema20) / ema20 * 100.0) if pd.notna(ema20) and ema20 > 0 else np.nan
    atr_pct = (atr14 / price * 100.0) if pd.notna(atr14) and price > 0 else np.nan

    range_pct_1d = range_pct(safe_float(last["High"], np.nan), safe_float(last["Low"], np.nan), price)
    range_pct_10d_avg = avg([range_pct(h, l, c) for h, l, c in zip(hist10["High"], hist10["Low"], hist10["Close"])])

    hist10_high = safe_float(hist10["High"].max(), np.nan)
    hist10_low = safe_float(hist10["Low"].min(), np.nan)
    base_tightness10 = ((hist10_high - hist10_low) / price * 100.0) if pd.notna(price) and price > 0 else np.nan

    low30 = safe_float(hist60["Low"].iloc[-30:].min() if len(hist60) >= 30 else hist60["Low"].min(), np.nan)
    high90 = safe_float(hist90["High"].max(), np.nan)
    drawdown90 = ((price / high90) - 1) * 100.0 if pd.notna(high90) and high90 > 0 else np.nan
    rebound30 = ((price - low30) / low30) * 100.0 if pd.notna(low30) and low30 > 0 else np.nan

    high20_ex_last = safe_float(hist20["High"].iloc[:-1].max() if len(hist20) > 1 else hist20["High"].max(), np.nan)
    breakout20 = 1 if pd.notna(high20_ex_last) and price > high20_ex_last else 0

    avg_range20 = max(avg([
        range_pct(h, l, c) for h, l, c in zip(hist20["High"].iloc[:-1], hist20["Low"].iloc[:-1], hist20["Close"].iloc[:-1])
    ]), 0.0001)
    range_expansion = range_pct_1d / avg_range20 if pd.notna(range_pct_1d) else np.nan

    features = {
        "price": price,
        "prev_day_ret": prev_day_ret,
        "prev_close_strength": prev_close_strength * 100 if pd.notna(prev_close_strength) else np.nan,
        "prev_vol_ratio": prev_vol_ratio,
        "prev_dollar_vol": prev_dollar_vol,
        "ema9_dist_pct": ema9_dist_pct,
        "ema20_dist_pct": ema20_dist_pct,
        "atr_pct": atr_pct,
        "range_pct_1d": range_pct_1d,
        "range_pct_10d_avg": range_pct_10d_avg,
        "base_tightness10": base_tightness10,
        "drawdown90": drawdown90,
        "rebound30": rebound30,
        "breakout20": breakout20,
        "range_expansion": range_expansion,
    }
    return features


# ============================================================
# ML LAYER
# ============================================================
def model_paths(model_name):
    return {
        "model": os.path.join(MODEL_DIR, f"{model_name}.joblib"),
        "meta": os.path.join(MODEL_DIR, f"{model_name}_meta.json"),
    }


def save_model_bundle(model_name, model, feature_cols, metrics):
    paths = model_paths(model_name)
    joblib.dump({"model": model, "feature_cols": feature_cols}, paths["model"])
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def load_model_bundle(model_name):
    paths = model_paths(model_name)
    if not os.path.exists(paths["model"]):
        return None, None, None
    payload = joblib.load(paths["model"])
    metrics = {}
    if os.path.exists(paths["meta"]):
        with open(paths["meta"], "r", encoding="utf-8") as f:
            metrics = json.load(f)
    return payload.get("model"), payload.get("feature_cols"), metrics


def train_random_forest_classifier(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)

    train_prob = clf.predict_proba(X_train)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1] if len(X_test) else np.array([])

    def prob_metrics(y_true, probs):
        if len(y_true) == 0:
            return {"count": 0, "positive_rate": None, "mean_prob": None}
        preds = (probs >= 0.5).astype(int)
        acc = float((preds == y_true).mean())
        precision = float(((preds == 1) & (y_true == 1)).sum() / max((preds == 1).sum(), 1))
        recall = float(((preds == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1))
        return {
            "count": int(len(y_true)),
            "positive_rate": round(float(y_true.mean()), 4),
            "mean_prob": round(float(np.mean(probs)), 4),
            "accuracy_0_5": round(acc, 4),
            "precision_0_5": round(precision, 4),
            "recall_0_5": round(recall, 4),
        }

    metrics = {
        "train": prob_metrics(y_train, train_prob),
        "test": prob_metrics(y_test, test_prob),
    }
    return clf, metrics


def build_training_rows_for_symbol(symbol, daily_df, model_type):
    rows = []
    if daily_df is None or daily_df.empty or len(daily_df) < 120:
        return rows

    temp = daily_df.copy()
    temp["NYDate"] = [local_date_from_iso(x) for x in temp.index]
    ny_dates = temp["NYDate"].tolist()

    for idx in range(100, len(temp) - 1):
        trade_date_str = ny_dates[idx]
        next_bar = temp.iloc[idx + 1]

        ctx = get_trade_date_context(temp.iloc[:idx + 1].copy(), trade_date_str)
        if ctx is None:
            continue

        feat = extract_snapshot_features_from_ctx(ctx)
        entry_ref = feat["price"]

        next_high = safe_float(next_bar["High"], np.nan)
        next_low = safe_float(next_bar["Low"], np.nan)

        if pd.isna(entry_ref) or entry_ref <= 0 or pd.isna(next_high) or pd.isna(next_low):
            continue

        # Conservative labels
        cont_tp = entry_ref * 1.06
        cont_stop = entry_ref * 0.95
        cont_label = int((next_high >= cont_tp) and not (next_low <= cont_stop and next_high < cont_tp))

        super_tp = entry_ref * 1.15
        super_stop = entry_ref * 0.88
        super_label = int(
            (entry_ref <= MAX_PRICE_SUPERNOVA) and
            (next_high >= super_tp) and
            not (next_low <= super_stop and next_high < super_tp)
        )

        base = {
            "symbol": symbol,
            "trade_date": trade_date_str,
            **feat
        }

        if model_type == "continuation":
            base["label"] = cont_label
            rows.append(base)
        else:
            base["label"] = super_label
            rows.append(base)

    return rows


def build_training_dataset(symbols, model_type):
    rows = []
    for symbol in symbols:
        daily_df = get_daily_df(symbol, lookback_days=520)
        rows.extend(build_training_rows_for_symbol(symbol, daily_df, model_type))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    feature_cols = CONT_FEATURE_COLS if model_type == "continuation" else SUPER_FEATURE_COLS
    keep = ["symbol", "trade_date", "label"] + feature_cols
    df = df[keep].copy()
    df = df.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)
    return df


def train_model_from_dataset(df, model_type):
    if df is None or df.empty:
        raise ValueError("Dataset boş.")

    feature_cols = CONT_FEATURE_COLS if model_type == "continuation" else SUPER_FEATURE_COLS
    df = df.sort_values("trade_date").reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    split_idx = max(split_idx, 1)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols].astype(float).values
    y_train = train_df["label"].astype(int).values
    X_test = test_df[feature_cols].astype(float).values if not test_df.empty else np.empty((0, len(feature_cols)))
    y_test = test_df["label"].astype(int).values if not test_df.empty else np.array([])

    model, metrics = train_random_forest_classifier(X_train, y_train, X_test, y_test)
    metrics["feature_cols"] = feature_cols
    metrics["rows"] = int(len(df))
    metrics["train_rows"] = int(len(train_df))
    metrics["test_rows"] = int(len(test_df))
    return model, feature_cols, metrics


def predict_model_probability(model_name, feature_dict):
    model, feature_cols, metrics = load_model_bundle(model_name)
    if model is None or not feature_cols:
        return np.nan, {}
    row = [safe_float(feature_dict.get(col, np.nan), np.nan) for col in feature_cols]
    if any(pd.isna(x) for x in row):
        return np.nan, metrics
    prob = float(model.predict_proba(np.array(row).reshape(1, -1))[0, 1])
    return prob, metrics


# ============================================================
# QUOTE / RISK
# ============================================================
def bid_ask_from_quote(quote):
    if not quote:
        return np.nan, np.nan
    bid = safe_float(quote.get("bp"), np.nan)
    ask = safe_float(quote.get("ap"), np.nan)
    return bid, ask


def spread_pct(bid, ask):
    if pd.isna(bid) or pd.isna(ask) or bid <= 0 or ask <= 0 or ask < bid:
        return np.nan
    mid = (bid + ask) / 2
    if mid <= 0:
        return np.nan
    return ((ask - bid) / mid) * 100.0


def late_entry_status(price, entry_idea):
    if pd.isna(price) or pd.isna(entry_idea) or entry_idea <= 0:
        return "UNKNOWN", np.nan
    ext = (price - entry_idea) / entry_idea
    if ext <= 0.03:
        return "NORMAL", ext * 100
    if ext <= DEFAULT_MAX_EXTENSION:
        return "PULLBACK_ONLY", ext * 100
    return "NO_FRESH_BUY", ext * 100


def risk_gate(decision, price, entry_idea, stop, bid, ask, pre_dollar_vol):
    late_status, ext_pct = late_entry_status(price, entry_idea)
    spr = spread_pct(bid, ask)

    risk_tags = []
    real_money_allowed = True

    if decision not in ["GÜÇLÜ AL", "AL"]:
        real_money_allowed = False
        risk_tags.append("decision_not_actionable")

    if late_status == "PULLBACK_ONLY":
        real_money_allowed = False
        risk_tags.append("late_entry_pullback_only")
    elif late_status == "NO_FRESH_BUY":
        real_money_allowed = False
        risk_tags.append("too_extended")

    if pd.notna(spr) and spr > 3.5:
        real_money_allowed = False
        risk_tags.append("wide_spread")

    if pd.notna(pre_dollar_vol) and pre_dollar_vol < DEFAULT_MIN_PRE_DOLLAR_VOL:
        real_money_allowed = False
        risk_tags.append("thin_liquidity")

    stop_dist_pct = np.nan
    if pd.notna(entry_idea) and pd.notna(stop) and entry_idea > 0:
        stop_dist_pct = ((entry_idea - stop) / entry_idea) * 100.0
        if stop_dist_pct > 18:
            real_money_allowed = False
            risk_tags.append("stop_too_wide")

    if pd.isna(bid) or pd.isna(ask):
        risk_tags.append("quote_unavailable")

    risk_grade = "LOW"
    if not real_money_allowed:
        risk_grade = "BLOCKED"
    elif decision == "GÜÇLÜ AL" and (pd.isna(spr) or spr <= 1.2):
        risk_grade = "MEDIUM"
    else:
        risk_grade = "ELEVATED"

    return {
        "Late_Status": late_status,
        "Extension_%": round_smart(ext_pct),
        "Bid": round_smart(bid),
        "Ask": round_smart(ask),
        "Spread_%": round_smart(spr),
        "Stop_Dist_%": round_smart(stop_dist_pct),
        "Risk_Grade": risk_grade,
        "Real_Money_Allowed": bool(real_money_allowed),
        "Risk_Tags": " | ".join(risk_tags)
    }


# ============================================================
# CONTINUATION ENGINE
# ============================================================
def score_continuation(last_bar, prev_bar, hist20, pre_ctx, pre_ratio, flags, cont_score_al, cont_score_strong, ml_prob=np.nan):
    notes = []
    score = 0

    prev_close = safe_float(last_bar["Close"], np.nan)
    prev_prev_close = safe_float(prev_bar["Close"], np.nan)
    prev_ret = ((prev_close - prev_prev_close) / prev_prev_close * 100.0) if prev_prev_close > 0 else np.nan
    prev_cs = closing_strength(prev_close, safe_float(last_bar["Low"], np.nan), safe_float(last_bar["High"], np.nan))

    avg_vol20 = max(avg(hist20["Volume"].iloc[:-1] if len(hist20) > 1 else hist20["Volume"]), 1)
    vol_ratio = safe_float(last_bar["Volume"], 0) / avg_vol20
    dollar_vol = safe_float(last_bar["Volume"], 0) * prev_close

    ema9 = calc_ema(hist20["Close"], 9).iloc[-1] if len(hist20) >= 9 else np.nan
    ema20 = calc_ema(hist20["Close"], 20).iloc[-1] if len(hist20) >= 20 else np.nan

    if prev_close < MIN_PRICE:
        notes.append("0.10 altı")
        return 0, "ALMA", notes, {}
    if prev_close > MAX_PRICE_CONTINUATION:
        notes.append("Fiyat çok yüksek")
        return 0, "ALMA", notes, {}

    if pd.notna(prev_ret):
        if 4 <= prev_ret < 12:
            score += 14
        elif 12 <= prev_ret < 35:
            score += 22
        elif 35 <= prev_ret < 70:
            score += 10
        elif prev_ret < 0:
            score -= 12

    if pd.notna(prev_cs):
        if prev_cs >= 0.85:
            score += 20
        elif prev_cs >= 0.72:
            score += 12
        elif prev_cs < 0.45:
            score -= 10
            notes.append("Weak close")

    if vol_ratio >= 1.5:
        score += 12
    if vol_ratio >= 3:
        score += 8
        notes.append("Volume expansion")

    if dollar_vol >= 500_000:
        score += 10
    elif dollar_vol < 100_000:
        score -= 10

    if pd.notna(ema9) and pd.notna(ema20) and prev_close > ema9 > ema20:
        score += 12
        notes.append("EMA9 > EMA20")
    elif pd.notna(ema20) and prev_close < ema20:
        score -= 8

    if pre_ctx["source"] == "REAL_PREMARKET":
        gap_pct = ((pre_ctx["price"] - prev_close) / prev_close * 100.0) if prev_close > 0 else np.nan
        pre_dollar_vol = pre_ctx["price"] * pre_ctx["pre_vol"] if pd.notna(pre_ctx["price"]) else np.nan
        above_vwap = pd.notna(pre_ctx["pre_vwap"]) and pd.notna(pre_ctx["price"]) and pre_ctx["price"] > pre_ctx["pre_vwap"]

        if pd.notna(gap_pct):
            if 2 <= gap_pct < 12:
                score += 12
            elif 12 <= gap_pct < 35:
                score += 16
            elif gap_pct >= 35:
                score += 6
                notes.append("Hot gap")

        if pd.notna(pre_ratio):
            if pre_ratio >= 0.8:
                score += 8
            if pre_ratio >= 1.5:
                score += 10

        if pd.notna(pre_dollar_vol):
            if pre_dollar_vol >= 250_000:
                score += 12
            elif pre_dollar_vol < DEFAULT_MIN_PRE_DOLLAR_VOL:
                score -= 12

        if above_vwap:
            score += 12
        else:
            score -= 10
            notes.append("Premarket VWAP altı")

        if pd.notna(pre_ctx["hold_quality"]):
            if pre_ctx["hold_quality"] >= 80:
                score += 16
            elif pre_ctx["hold_quality"] >= 65:
                score += 8
            elif pre_ctx["hold_quality"] < 45:
                score -= 12
    else:
        gap_pct = np.nan
        pre_dollar_vol = np.nan
        above_vwap = False
        notes.append("Premarket veri yok")

    if flags["recent_reverse_split"]:
        score -= 25
        notes.append("Recent reverse split")
    if flags["otc_risk"]:
        score -= 30
        notes.append("OTC risk")
    if flags["recent_deficiency"]:
        score -= 6
        notes.append("Deficiency")

    if pd.notna(ml_prob):
        ml_score = ml_prob * 100.0
        score = 0.70 * score + 0.30 * ml_score
        notes.append(f"ML={round(ml_score,1)}")

    score = clamp(round(score), 0, 100)

    if pre_ctx["source"] != "REAL_PREMARKET":
        decision = "İZLE" if score >= cont_score_al - 4 else "ALMA"
    elif (
        score >= cont_score_strong and above_vwap and
        pd.notna(pre_ctx["hold_quality"]) and pre_ctx["hold_quality"] >= 70 and
        pd.notna(pre_dollar_vol) and pre_dollar_vol >= 300_000
    ):
        decision = "GÜÇLÜ AL"
    elif (
        score >= cont_score_al and above_vwap and
        pd.notna(pre_ctx["hold_quality"]) and pre_ctx["hold_quality"] >= 58
    ):
        decision = "AL"
    elif score >= cont_score_al - 4:
        decision = "İZLE"
    else:
        decision = "ALMA"

    return score, decision, notes, {
        "prev_ret": prev_ret,
        "prev_cs": prev_cs,
        "vol_ratio": vol_ratio,
        "dollar_vol": dollar_vol,
        "ema9": ema9,
        "ema20": ema20,
        "gap_pct": gap_pct,
        "pre_dollar_vol": pre_dollar_vol,
        "pre_vr": pre_ratio,
        "above_vwap": above_vwap,
        "ml_prob": ml_prob
    }


def build_continuation_row(symbol, daily_df, minute_df, trade_date_str, cutoff_time, cont_score_al, cont_score_strong, quotes_map):
    ctx = get_trade_date_context(daily_df, trade_date_str)
    if ctx is None:
        return None

    flags = get_flags(symbol)
    feat = extract_snapshot_features_from_ctx(ctx)
    ml_prob, _ = predict_model_probability("continuation_model", feat)

    pre_ctx = get_premarket_context(minute_df, trade_date_str, cutoff_time)
    baseline_median, _, samples = premarket_baseline_ratio(minute_df, ctx["prior_dates"], trade_date_str, cutoff_time)

    pre_ratio = np.nan
    if pre_ctx["source"] == "REAL_PREMARKET" and pd.notna(baseline_median) and baseline_median > 0:
        pre_ratio = pre_ctx["pre_vol"] / baseline_median

    score, decision, notes, extra = score_continuation(
        ctx["last"], ctx["prev"], ctx["hist20"], pre_ctx, pre_ratio, flags,
        cont_score_al, cont_score_strong, ml_prob=ml_prob
    )

    prev_high = safe_float(ctx["last"]["High"], np.nan)
    prev_close = safe_float(ctx["last"]["Close"], np.nan)
    gap_pct = extra.get("gap_pct", np.nan)

    entry_idea = np.nan
    entry_type = "NO_TRADE"
    stop = np.nan
    tp1 = np.nan
    tp2 = np.nan

    if decision in ["GÜÇLÜ AL", "AL", "İZLE"]:
        if pre_ctx["source"] == "REAL_PREMARKET":
            reclaim = prev_high * 1.01 if pd.notna(prev_high) else np.nan
            vwap_entry = pre_ctx["pre_vwap"] * 1.01 if pd.notna(pre_ctx["pre_vwap"]) else np.nan
            entry_idea = max(reclaim, vwap_entry) if pd.notna(reclaim) and pd.notna(vwap_entry) else (reclaim if pd.notna(reclaim) else vwap_entry)

            if pd.notna(entry_idea) and pd.notna(pre_ctx["price"]):
                ext = (pre_ctx["price"] - entry_idea) / entry_idea if entry_idea > 0 else np.nan
                if pd.notna(ext) and ext <= 0.03:
                    entry_type = "BUY_NEAR_ENTRY"
                elif pd.notna(ext) and ext <= DEFAULT_MAX_EXTENSION:
                    entry_type = "WAIT_RETEST"
                else:
                    entry_type = "TOO_EXTENDED"
        else:
            entry_idea = prev_high * 1.01 if pd.notna(prev_high) else np.nan
            entry_type = "WATCH_RECLAIM"

        if pd.notna(entry_idea) and entry_idea > 0:
            stop = entry_idea * 0.93
            tp1 = entry_idea * 1.10
            tp2 = entry_idea * 1.15

    quote = quotes_map.get(symbol, {})
    bid, ask = bid_ask_from_quote(quote)
    risk = risk_gate(decision, pre_ctx["price"], entry_idea, stop, bid, ask, extra.get("pre_dollar_vol", np.nan))

    return {
        "Engine": "CONTINUATION",
        "Symbol": symbol,
        "Decision": decision,
        "Score": score,
        "ML_Prob": round_smart(ml_prob * 100 if pd.notna(ml_prob) else np.nan),
        "Price": round_smart(pre_ctx["price"]),
        "Prev_Close": round_smart(prev_close),
        "Gap_%": round_smart(gap_pct),
        "Prev_Ret_%": round_smart(extra.get("prev_ret", np.nan)),
        "Prev_Close_Strength": round_smart(extra.get("prev_cs", np.nan) * 100 if pd.notna(extra.get("prev_cs", np.nan)) else np.nan),
        "Prev_Vol_Ratio": round_smart(extra.get("vol_ratio", np.nan)),
        "EMA9": round_smart(extra.get("ema9", np.nan)),
        "EMA20": round_smart(extra.get("ema20", np.nan)),
        "Pre_Vol": safe_int(pre_ctx["pre_vol"], 0),
        "Pre_Baseline": safe_int(baseline_median, 0) if pd.notna(baseline_median) else 0,
        "Pre_Vol_Ratio": round_smart(pre_ratio),
        "Pre_VWAP": round_smart(pre_ctx["pre_vwap"]),
        "Hold_%": round_smart(pre_ctx["hold_quality"]),
        "Pre_Dollar_Vol": round_smart(extra.get("pre_dollar_vol", np.nan)),
        "Source": pre_ctx["source"],
        "Entry_Type": entry_type,
        "Entry_Idea": round_smart(entry_idea),
        "Stop": round_smart(stop),
        "TP1": round_smart(tp1),
        "TP2": round_smart(tp2),
        "Notes": " | ".join(notes),
        "Samples": samples,
        **risk
    }


# ============================================================
# SUPERNOVA ENGINE
# ============================================================
ROCKET_PROTOTYPES = [
    {
        "name": "RMSG_STYLE_SUPERNOVA",
        "weights": {
            "price": 10, "drawdown90": 8, "base_tightness10": 8, "prev_day_ret": 9,
            "prev_vol_ratio": 10, "prev_close_strength": 10, "breakout20": 6,
            "gap_pct": 8, "pre_vol_ratio": 10, "hold_quality": 10,
            "pre_dollar_vol": 8, "above_prev_high": 3
        },
        "bands": {
            "price": (0.10, 3.50, 0.05, 5.00),
            "drawdown90": (-90, -25, -99, -5),
            "base_tightness10": (5, 35, 0, 80),
            "prev_day_ret": (5, 45, -10, 90),
            "prev_vol_ratio": (2, 15, 0.5, 40),
            "prev_close_strength": (75, 100, 45, 100),
            "breakout20": (1, 1, 0, 1),
            "gap_pct": (10, 80, -10, 150),
            "pre_vol_ratio": (1.2, 10, 0.2, 30),
            "hold_quality": (65, 100, 40, 100),
            "pre_dollar_vol": (200000, 20000000, 50000, 80000000),
            "above_prev_high": (1, 1, 0, 1)
        }
    },
    {
        "name": "SQUEEZE_RECLAIM_STYLE",
        "weights": {
            "price": 9, "drawdown90": 8, "base_tightness10": 9, "prev_day_ret": 10,
            "prev_vol_ratio": 10, "prev_close_strength": 10, "breakout20": 7,
            "gap_pct": 7, "pre_vol_ratio": 8, "hold_quality": 9,
            "pre_dollar_vol": 8, "above_prev_high": 5
        },
        "bands": {
            "price": (0.25, 5.00, 0.10, 7.00),
            "drawdown90": (-80, -20, -99, 0),
            "base_tightness10": (3, 25, 0, 60),
            "prev_day_ret": (12, 60, -5, 120),
            "prev_vol_ratio": (2, 12, 0.5, 35),
            "prev_close_strength": (78, 100, 50, 100),
            "breakout20": (1, 1, 0, 1),
            "gap_pct": (5, 50, -10, 100),
            "pre_vol_ratio": (1.0, 8, 0.2, 20),
            "hold_quality": (70, 100, 45, 100),
            "pre_dollar_vol": (150000, 15000000, 40000, 60000000),
            "above_prev_high": (1, 1, 0, 1)
        }
    },
]


def score_structural(price, drawdown90, base_tightness10, rebound30, breakout20, flags):
    score = 0
    notes = []

    if flags["otc_risk"]:
        return 0, ["OTC risk"], True
    if price < MIN_PRICE:
        return 0, ["0.10 altı"], True
    if price > MAX_PRICE_SUPERNOVA:
        return 0, ["5 dolar üstü"], True

    if 0.10 <= price <= 1:
        score += 24
    elif 1 < price <= 3:
        score += 20
    elif 3 < price <= 5:
        score += 12

    if -90 <= drawdown90 <= -35:
        score += 18
    elif -35 < drawdown90 <= -10:
        score += 8

    if 4 <= base_tightness10 <= 28:
        score += 16
    elif 28 < base_tightness10 <= 50:
        score += 8
    elif base_tightness10 > 70:
        score -= 8
        notes.append("Loose base")

    if 5 <= rebound30 <= 120:
        score += 10
    elif rebound30 > 200:
        score -= 5

    if breakout20:
        score += 12
        notes.append("20d reclaim")

    if flags["recent_reverse_split"]:
        score -= 28
        notes.append("Recent reverse split")
    if flags["recent_deficiency"]:
        score -= 6
        notes.append("Deficiency")

    return clamp(round(score), 0, 100), notes, False


def score_ignition(prev_day_ret, prev_close_strength, prev_vol_ratio, prev_dollar_vol, range_expansion, flags):
    score = 0
    notes = []

    if 4 <= prev_day_ret < 15:
        score += 16
    elif 15 <= prev_day_ret < 45:
        score += 22
    elif 45 <= prev_day_ret < 100:
        score += 14
    elif prev_day_ret < 0:
        score -= 12

    if 1.5 <= prev_vol_ratio < 3:
        score += 12
    elif 3 <= prev_vol_ratio < 8:
        score += 20
    elif prev_vol_ratio >= 8:
        score += 24
        notes.append("Vol shock")
    elif prev_vol_ratio < 0.8:
        score -= 8

    if prev_close_strength >= 80:
        score += 18
    elif prev_close_strength >= 65:
        score += 10
    elif prev_close_strength < 45:
        score -= 10
        notes.append("Weak close")

    if 150_000 <= prev_dollar_vol < 600_000:
        score += 10
    elif 600_000 <= prev_dollar_vol < 3_000_000:
        score += 16
    elif prev_dollar_vol >= 3_000_000:
        score += 20
    elif prev_dollar_vol < 50_000:
        score -= 10

    if 1.3 <= range_expansion < 2.5:
        score += 8
    elif range_expansion >= 2.5:
        score += 14

    if flags["catalyst_fresh"]:
        score += 12
        notes.append("Fresh catalyst flag")

    return clamp(round(score), 0, 100), notes


def score_premarket_rocket(gap_pct, pre_vol_ratio, pre_dollar_vol, hold_quality, above_vwap, above_prev_high, pre_range_pct, source):
    score = 0
    notes = []

    if source != "REAL_PREMARKET":
        return 0, ["Premarket veri yok"], False

    if 3 <= gap_pct < 15:
        score += 12
    elif 15 <= gap_pct < 50:
        score += 20
    elif 50 <= gap_pct < 150:
        score += 12
        notes.append("Aşırı sıcak gap")
    elif gap_pct < 0:
        score -= 10

    if 0.8 <= pre_vol_ratio < 2:
        score += 10
    elif 2 <= pre_vol_ratio < 6:
        score += 20
    elif pre_vol_ratio >= 6:
        score += 24
        notes.append("Premkt vol shock")
    elif pre_vol_ratio < 0.4:
        score -= 8

    if 150_000 <= pre_dollar_vol < 500_000:
        score += 12
    elif 500_000 <= pre_dollar_vol < 2_000_000:
        score += 18
    elif pre_dollar_vol >= 2_000_000:
        score += 22
    elif pre_dollar_vol < DEFAULT_MIN_PRE_DOLLAR_VOL:
        score -= 12

    if hold_quality >= 80:
        score += 18
    elif hold_quality >= 65:
        score += 10
    elif hold_quality < 45:
        score -= 12

    if above_vwap:
        score += 14
    else:
        score -= 12
        notes.append("VWAP altı")

    if above_prev_high:
        score += 10
        notes.append("Prev high reclaim")

    if pd.notna(pre_range_pct) and pre_range_pct > 60:
        score -= 8
        notes.append("Range çok geniş")

    hard_reject = (not above_vwap) or (hold_quality < 45) or (pre_dollar_vol < DEFAULT_MIN_PRE_DOLLAR_VOL)
    return clamp(round(score), 0, 100), notes, hard_reject


def similarity_band(value, ideal_low, ideal_high, hard_low, hard_high):
    if pd.isna(value):
        return 0.0
    value = float(value)
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < hard_low or value > hard_high:
        return 0.0
    if value < ideal_low:
        return (value - hard_low) / (ideal_low - hard_low)
    return (hard_high - value) / (hard_high - ideal_high)


def compute_pattern_similarity(feature_dict):
    results = []
    for proto in ROCKET_PROTOTYPES:
        weighted = 0.0
        total = 0.0
        for key, weight in proto["weights"].items():
            ideal_low, ideal_high, hard_low, hard_high = proto["bands"][key]
            val = feature_dict.get(key, np.nan)
            if isinstance(val, bool):
                val = 1 if val else 0
            sim = similarity_band(val, ideal_low, ideal_high, hard_low, hard_high)
            weighted += sim * weight
            total += weight
        score = (weighted / total * 100.0) if total > 0 else 0.0
        results.append((proto["name"], round(score)))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[0][0], results[0][1], results[:3]


def build_supernova_row(symbol, daily_df, minute_df, trade_date_str, cutoff_time, quotes_map, super_pattern_al, super_pattern_strong):
    ctx = get_trade_date_context(daily_df, trade_date_str)
    if ctx is None:
        return None

    flags = get_flags(symbol)
    feat = extract_snapshot_features_from_ctx(ctx)
    ml_prob, _ = predict_model_probability("supernova_model", feat)

    last = ctx["last"]
    price = safe_float(last["Close"], np.nan)

    structural_score, structural_notes, struct_reject = score_structural(
        feat["price"], feat["drawdown90"], feat["base_tightness10"], feat["rebound30"], bool(feat["breakout20"]), flags
    )
    ignition_score, ignition_notes = score_ignition(
        feat["prev_day_ret"], feat["prev_close_strength"], feat["prev_vol_ratio"],
        feat["prev_dollar_vol"], feat["range_expansion"], flags
    )

    pre_ctx = get_premarket_context(minute_df, trade_date_str, cutoff_time)
    baseline_median, _, _ = premarket_baseline_ratio(minute_df, ctx["prior_dates"], trade_date_str, cutoff_time)
    pre_vol_ratio = pre_ctx["pre_vol"] / baseline_median if pd.notna(baseline_median) and baseline_median > 0 else np.nan
    gap_pct = ((pre_ctx["price"] - price) / price * 100.0) if pd.notna(pre_ctx["price"]) and price > 0 else np.nan
    pre_dollar_vol = pre_ctx["price"] * pre_ctx["pre_vol"] if pd.notna(pre_ctx["price"]) else np.nan
    above_vwap = pd.notna(pre_ctx["pre_vwap"]) and pd.notna(pre_ctx["price"]) and pre_ctx["price"] > pre_ctx["pre_vwap"]
    above_prev_high = pd.notna(pre_ctx["price"]) and pd.notna(last["High"]) and pre_ctx["price"] > safe_float(last["High"], np.nan)

    premarket_score, pre_notes, pre_reject = score_premarket_rocket(
        safe_float(gap_pct, 0), safe_float(pre_vol_ratio, 0), safe_float(pre_dollar_vol, 0),
        safe_float(pre_ctx["hold_quality"], 0), bool(above_vwap), bool(above_prev_high),
        safe_float(pre_ctx["pre_range_pct"], np.nan), pre_ctx["source"]
    )

    pattern_name, pattern_score, top_matches = compute_pattern_similarity({
        "price": feat["price"],
        "drawdown90": feat["drawdown90"],
        "base_tightness10": feat["base_tightness10"],
        "prev_day_ret": feat["prev_day_ret"],
        "prev_vol_ratio": feat["prev_vol_ratio"],
        "prev_close_strength": feat["prev_close_strength"],
        "breakout20": feat["breakout20"],
        "gap_pct": gap_pct,
        "pre_vol_ratio": pre_vol_ratio,
        "hold_quality": pre_ctx["hold_quality"],
        "pre_dollar_vol": pre_dollar_vol,
        "above_prev_high": 1 if above_prev_high else 0
    })

    if pd.notna(ml_prob):
        pattern_score = round(0.70 * pattern_score + 0.30 * (ml_prob * 100.0))
        pre_notes.append(f"ML={round(ml_prob*100,1)}")

    decision = "ALMA"
    if not struct_reject and not pre_reject:
        composite = np.mean([structural_score, ignition_score, premarket_score, pattern_score])
        if (
            structural_score >= 50 and ignition_score >= 60 and
            premarket_score >= 72 and pattern_score >= super_pattern_strong and composite >= 72
        ):
            decision = "GÜÇLÜ AL"
        elif (
            structural_score >= 40 and ignition_score >= 50 and
            premarket_score >= 58 and pattern_score >= super_pattern_al and composite >= 60
        ):
            decision = "AL"
        elif np.mean([structural_score, ignition_score, pattern_score]) >= 65:
            decision = "İZLE"
    else:
        if np.mean([structural_score, ignition_score, pattern_score]) >= 68:
            decision = "İZLE"

    entry_type = "NO_TRADE"
    entry_idea = np.nan
    stop = np.nan
    tp1 = np.nan
    tp2 = np.nan

    if decision in ["GÜÇLÜ AL", "AL", "İZLE"]:
        reclaim = safe_float(last["High"], np.nan) * 1.01 if pd.notna(last["High"]) else np.nan
        vwap_entry = pre_ctx["pre_vwap"] * 1.01 if pd.notna(pre_ctx["pre_vwap"]) else np.nan
        entry_idea = max(reclaim, vwap_entry) if pd.notna(reclaim) and pd.notna(vwap_entry) else (reclaim if pd.notna(reclaim) else vwap_entry)

        if pd.notna(pre_ctx["price"]) and pd.notna(entry_idea) and entry_idea > 0:
            ext = (pre_ctx["price"] - entry_idea) / entry_idea
            if ext <= 0.04:
                entry_type = "BUY_NEAR_ENTRY"
            elif ext <= 0.12:
                entry_type = "WAIT_RETEST"
            else:
                entry_type = "TOO_EXTENDED"
        else:
            entry_type = "WATCH_RECLAIM"

        if pd.notna(entry_idea) and entry_idea > 0:
            stop = entry_idea * 0.88
            tp1 = entry_idea * 1.15
            tp2 = entry_idea * 1.30

    quote = quotes_map.get(symbol, {})
    bid, ask = bid_ask_from_quote(quote)
    risk = risk_gate(decision, pre_ctx["price"], entry_idea, stop, bid, ask, pre_dollar_vol)

    return {
        "Engine": "SUPERNOVA",
        "Symbol": symbol,
        "Decision": decision,
        "Structural_Score": structural_score,
        "Ignition_Score": ignition_score,
        "Premarket_Score": premarket_score,
        "Pattern_Score": pattern_score,
        "ML_Prob": round_smart(ml_prob * 100 if pd.notna(ml_prob) else np.nan),
        "Best_Pattern": pattern_name,
        "Price": round_smart(pre_ctx["price"]),
        "Prev_Close": round_smart(price),
        "Gap_%": round_smart(gap_pct),
        "Prev_Ret_%": round_smart(feat["prev_day_ret"]),
        "Prev_Close_Strength": round_smart(feat["prev_close_strength"]),
        "Prev_Vol_Ratio": round_smart(feat["prev_vol_ratio"]),
        "Prev_Dollar_Vol": round_smart(feat["prev_dollar_vol"]),
        "Drawdown90_%": round_smart(feat["drawdown90"]),
        "Base_Tightness10_%": round_smart(feat["base_tightness10"]),
        "Breakout20": bool(feat["breakout20"]),
        "Pre_Vol": safe_int(pre_ctx["pre_vol"], 0),
        "Pre_Baseline": safe_int(baseline_median, 0) if pd.notna(baseline_median) else 0,
        "Pre_Vol_Ratio": round_smart(pre_vol_ratio),
        "Pre_VWAP": round_smart(pre_ctx["pre_vwap"]),
        "Hold_%": round_smart(pre_ctx["hold_quality"]),
        "Pre_Dollar_Vol": round_smart(pre_dollar_vol),
        "Source": pre_ctx["source"],
        "Entry_Type": entry_type,
        "Entry_Idea": round_smart(entry_idea),
        "Stop": round_smart(stop),
        "TP1": round_smart(tp1),
        "TP2": round_smart(tp2),
        "Top_Matches": " | ".join([f"{n}:{s}" for n, s in top_matches]),
        "Notes": " | ".join(structural_notes + ignition_notes + pre_notes),
        **risk
    }


# ============================================================
# BUILD / RADAR
# ============================================================
def build_engine_rows(symbols, engine_name, trade_date_str=None, cutoff_time=None,
                      cont_score_al=DEFAULT_CONT_SCORE_AL, cont_score_strong=DEFAULT_CONT_SCORE_STRONG,
                      super_pattern_al=DEFAULT_SUPER_PATTERN_AL, super_pattern_strong=DEFAULT_SUPER_PATTERN_STRONG):
    rows = []
    live_mode = trade_date_str is None

    if live_mode:
        trade_date_str = ny_date_str()
        session = get_session_label()
        now_str = ny_time_str()
        if session == "premarket":
            cutoff_time = cutoff_time or min(now_str, "09:25:00")
        elif session == "regular":
            cutoff_time = "09:25:00"
        elif session in ["afterhours", "closed", "weekend"]:
            cutoff_time = None
        else:
            cutoff_time = None
    else:
        session = "backtest"
        cutoff_time = cutoff_time or "09:25:00"

    quotes_map = fetch_alpaca_latest_quotes(symbols) if have_alpaca() else {}
    daily_lookback = 520
    minute_lookback_days = 12

    for symbol in symbols:
        daily_df = get_daily_df(symbol, lookback_days=daily_lookback)

        if cutoff_time is None:
            minute_df = pd.DataFrame()
        else:
            end_dt = datetime.strptime(trade_date_str + " 23:59:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=NY_TZ).astimezone(UTC_TZ)
            start_dt = end_dt - timedelta(days=minute_lookback_days)
            minute_df = fetch_minute_df(symbol, start_dt, end_dt, timeframe="1Min")
            if minute_df.empty and have_alpaca():
                minute_df = fetch_minute_df(symbol, start_dt, end_dt, timeframe="5Min")

        if engine_name == "continuation":
            row = build_continuation_row(
                symbol, daily_df, minute_df, trade_date_str, cutoff_time or "09:25:00",
                cont_score_al, cont_score_strong, quotes_map
            )
        else:
            row = build_supernova_row(
                symbol, daily_df, minute_df, trade_date_str, cutoff_time or "09:25:00",
                quotes_map, super_pattern_al, super_pattern_strong
            )

        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, session, cutoff_time

    df = apply_session_overlay(df, session)

    if engine_name == "continuation":
        df["_rank"] = df["Decision"].map(decision_sort_key)
        df = df.sort_values(["_rank", "Score"], ascending=[True, False]).drop(columns=["_rank"]).reset_index(drop=True)
    else:
        df["_rank"] = df["Decision"].map(decision_sort_key)
        df = df.sort_values(["_rank", "Pattern_Score", "Premarket_Score", "Ignition_Score"],
                            ascending=[True, False, False, False]).drop(columns=["_rank"]).reset_index(drop=True)

    return df, session, cutoff_time


def build_radar(symbols=None, cont_symbols=None, super_symbols=None, cont_al=DEFAULT_CONT_SCORE_AL, cont_strong=DEFAULT_CONT_SCORE_STRONG, super_al=DEFAULT_SUPER_PATTERN_AL, super_strong=DEFAULT_SUPER_PATTERN_STRONG):
    cont_symbols = cont_symbols if cont_symbols is not None else (symbols or [])
    super_symbols = super_symbols if super_symbols is not None else (symbols or [])

    cont_df, session, cutoff = build_engine_rows(
        cont_symbols, "continuation", trade_date_str=None,
        cont_score_al=cont_al, cont_score_strong=cont_strong,
        super_pattern_al=super_al, super_pattern_strong=super_strong
    )
    super_df, _, _ = build_engine_rows(
        super_symbols, "supernova", trade_date_str=None,
        cont_score_al=cont_al, cont_score_strong=cont_strong,
        super_pattern_al=super_al, super_pattern_strong=super_strong
    )

    if cont_df.empty and super_df.empty:
        return pd.DataFrame(), session, cutoff

    cont_cols = ["Symbol", "Decision", "Score", "ML_Prob", "Entry_Idea", "Risk_Grade", "Real_Money_Allowed", "Trade_Phase", "Action_Status"]
    super_cols = ["Symbol", "Decision", "Pattern_Score", "ML_Prob", "Entry_Idea", "Risk_Grade", "Real_Money_Allowed", "Trade_Phase", "Action_Status"]

    cont_small = cont_df[cont_cols].copy() if not cont_df.empty else pd.DataFrame(columns=cont_cols)
    cont_small = cont_small.rename(columns={
        "Decision": "Cont_Decision",
        "Score": "Cont_Score",
        "ML_Prob": "Cont_ML_Prob",
        "Entry_Idea": "Cont_Entry",
        "Risk_Grade": "Cont_Risk",
        "Real_Money_Allowed": "Cont_Allowed",
        "Trade_Phase": "Cont_Phase",
        "Action_Status": "Cont_Action"
    })

    super_small = super_df[super_cols].copy() if not super_df.empty else pd.DataFrame(columns=super_cols)
    super_small = super_small.rename(columns={
        "Decision": "Super_Decision",
        "Pattern_Score": "Super_Pattern",
        "ML_Prob": "Super_ML_Prob",
        "Entry_Idea": "Super_Entry",
        "Risk_Grade": "Super_Risk",
        "Real_Money_Allowed": "Super_Allowed",
        "Trade_Phase": "Super_Phase",
        "Action_Status": "Super_Action"
    })

    merged = pd.merge(cont_small, super_small, on="Symbol", how="outer")

    def final_pick(row):
        cont_dec = row.get("Cont_Decision", None)
        super_dec = row.get("Super_Decision", None)
        if super_dec == "GÜÇLÜ AL":
            return "SUPERNOVA_GÜÇLÜ_AL"
        if cont_dec == "GÜÇLÜ AL":
            return "CONTINUATION_GÜÇLÜ_AL"
        if super_dec == "AL":
            return "SUPERNOVA_AL"
        if cont_dec == "AL":
            return "CONTINUATION_AL"
        if super_dec == "İZLE" or cont_dec == "İZLE":
            return "İZLE"
        return "ALMA"

    merged["Radar_Pick"] = merged.apply(final_pick, axis=1)
    merged["_rank"] = merged["Radar_Pick"].map({
        "SUPERNOVA_GÜÇLÜ_AL": 0,
        "CONTINUATION_GÜÇLÜ_AL": 1,
        "SUPERNOVA_AL": 2,
        "CONTINUATION_AL": 3,
        "İZLE": 4,
        "ALMA": 5
    }).fillna(9)
    merged = merged.sort_values("_rank").drop(columns=["_rank"]).reset_index(drop=True)
    return merged, session, cutoff


# ============================================================
# JOURNAL
# ============================================================
def render_journal_tab():
    st.subheader("🧾 Paper Journal / Manual Tracker")
    st.caption("Ayrı CSV log tutman yine en sağlıklısıdır.")
    journal_file = st.file_uploader("Journal CSV yükle (opsiyonel)", type=["csv"], key="journal_csv")
    if journal_file is not None:
        try:
            df = pd.read_csv(journal_file)
            st.dataframe(df, use_container_width=True)
            if "PnL" in df.columns:
                pnl = pd.to_numeric(df["PnL"], errors="coerce").fillna(0).sum()
                st.metric("Toplam PnL", round(float(pnl), 2))
        except Exception as e:
            st.error(f"CSV okunamadı: {e}")

    st.code("Date,Symbol,Engine,Decision,Entry,Exit,Stop,TP1,TP2,Shares,PnL,Notes", language="text")


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.success("ML Final mode aktif")

    st.header("Alpaca API")
    api_key_input = st.text_input("API Key ID", value=ALPACA_API_KEY, type="password")
    secret_key_input = st.text_input("Secret Key", value=ALPACA_SECRET_KEY, type="password")
    if api_key_input and secret_key_input:
        ALPACA_API_KEY = api_key_input
        ALPACA_SECRET_KEY = secret_key_input

    st.caption(f"Feed: {ALPACA_FEED}")
    st.divider()

    st.header("Thresholds")
    CONT_SCORE_AL = st.slider("Continuation AL skoru", 40, 90, DEFAULT_CONT_SCORE_AL, 1)
    CONT_SCORE_STRONG = st.slider("Continuation GÜÇLÜ AL skoru", 50, 95, DEFAULT_CONT_SCORE_STRONG, 1)
    SUPER_PATTERN_AL = st.slider("Supernova Pattern AL", 40, 90, DEFAULT_SUPER_PATTERN_AL, 1)
    SUPER_PATTERN_STRONG = st.slider("Supernova Pattern GÜÇLÜ AL", 50, 95, DEFAULT_SUPER_PATTERN_STRONG, 1)
    SHOW_ONLY_TRADEABLE = st.checkbox("Sadece AL / GÜÇLÜ AL / İZLE göster", value=False)
    N_SHOW = st.slider("Gösterilecek satır", 10, 200, 50, 5)


# ============================================================
# TABS
# ============================================================
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🧭 Radar",
    "📈 Continuation Engine",
    "🚀 Supernova Engine",
    "🧠 ML Trainer",
    "🛡️ Risk Monitor",
    "🧾 Journal",
    "🌙 Overnight Engine",
    "⚡ Premarket Ignition"
])

with tab0:
    st.subheader("Radar")
    st.caption("Aynı sembol havuzunu continuation ve supernova açısından tarar; birleşik öncelik listesi üretir.")

    radar_use_full_market = st.checkbox("NASDAQ + NYSE tam evreni tara", value=False, key="radar_use_full_market")
    if radar_use_full_market:
        st.info("Radar tam evren modu ilk çalıştırmada yavaş olabilir. Başlangıç için Evren=1500, Ön filtre=120, Detaylı scan=50 önerilir.")
    radar_symbols_text = st.text_area(
        "Radar sembol listesi",
        value=",".join(list(dict.fromkeys(DEFAULT_CONTINUATION_SYMBOLS + DEFAULT_SUPERNOVA_SYMBOLS))),
        height=100,
        key="radar_symbols"
    )

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        radar_exchanges = st.multiselect("Borsalar", ["NASDAQ", "NYSE"], default=["NASDAQ", "NYSE"], key="radar_exchanges")
    with r2:
        radar_prefilter_limit = st.slider("Ön filtre", 50, 500, 180, 10, key="radar_prefilter_limit")
    with r3:
        radar_deep_scan_limit = st.slider("Detaylı scan", 20, 250, 80, 10, key="radar_deep_scan_limit")
    with r4:
        radar_universe_limit = st.number_input("Evren sınırı (0=tamı / yavaş)", min_value=0, value=1500, step=500, key="radar_universe_limit")

    radar_run = st.button("Radar çalıştır", key="radar_run")

    if radar_run:
        if radar_use_full_market and not have_alpaca():
            st.warning("Tam evren taraması için Alpaca API gerekir.")
        else:
            if radar_use_full_market:
                cont_syms, cont_meta, cont_pre_df = resolve_scan_symbols(
                    radar_symbols_text, True, "continuation", ny_date_str(), radar_exchanges,
                    radar_prefilter_limit, radar_deep_scan_limit, radar_universe_limit
                )
                super_syms, super_meta, super_pre_df = resolve_scan_symbols(
                    radar_symbols_text, True, "supernova", ny_date_str(), radar_exchanges,
                    radar_prefilter_limit, radar_deep_scan_limit, radar_universe_limit
                )
                syms = sorted(list(dict.fromkeys(cont_syms + super_syms)))
            else:
                syms = parse_symbols(radar_symbols_text)
                cont_syms = syms
                super_syms = syms
                cont_meta = {"source": "manual", "universe_count": len(syms), "ranked_count": len(syms), "deep_count": len(syms)}
                super_meta = cont_meta
                cont_pre_df = pd.DataFrame()
                super_pre_df = pd.DataFrame()

            if not syms:
                st.warning("Taranacak sembol bulunamadı.")
            else:
                with st.spinner("Radar çalışıyor..."):
                    radar_df, session, cutoff = build_radar(
                        cont_symbols=cont_syms,
                        super_symbols=super_syms,
                        cont_al=CONT_SCORE_AL,
                        cont_strong=CONT_SCORE_STRONG,
                        super_al=SUPER_PATTERN_AL,
                        super_strong=SUPER_PATTERN_STRONG
                    )

                st.write(f"**Session:** {session} | **Cutoff:** {cutoff or '-'}")
                render_session_message(session)
                if radar_use_full_market:
                    st.info(
                        f"Continuation evreni: {cont_meta['universe_count']} | ranked: {cont_meta['ranked_count']} | deep scan: {cont_meta['deep_count']}  \n"
                        f"Supernova evreni: {super_meta['universe_count']} | ranked: {super_meta['ranked_count']} | deep scan: {super_meta['deep_count']}"
                    )

                if radar_df.empty:
                    st.warning("Radar sonucu yok.")
                else:
                    if SHOW_ONLY_TRADEABLE:
                        radar_df = radar_df[radar_df["Radar_Pick"] != "ALMA"].copy()
                    st.dataframe(radar_df.head(N_SHOW), use_container_width=True)
                    csv = radar_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 Radar CSV indir",
                        data=csv,
                        file_name=f"radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_radar"
                    )

                    if radar_use_full_market:
                        with st.expander("Ön filtre adayları"):
                            ctab1, ctab2 = st.tabs(["Continuation Prefilter", "Supernova Prefilter"])
                            with ctab1:
                                st.dataframe(cont_pre_df.head(min(N_SHOW, len(cont_pre_df))), use_container_width=True)
                            with ctab2:
                                st.dataframe(super_pre_df.head(min(N_SHOW, len(super_pre_df))), use_container_width=True)

with tab1:
    st.subheader("Continuation Engine")
    st.caption("Amaç: önceki gün güçlü kapanan ve ertesi gün devam etme ihtimali olan hisseleri bulmak.")
    col1, col2 = st.columns([2, 1])

    with col1:
        cont_symbols_text = st.text_area(
            "Sembol listesi",
            value=",".join(DEFAULT_CONTINUATION_SYMBOLS),
            height=100,
            key="cont_symbols"
        )
    with col2:
        cont_mode = st.radio("Mod", ["Canlı", "Backtest"], index=0, key="cont_mode")
        cont_date = st.date_input("Backtest tarihi", value=now_ny().date(), key="cont_date")
        cont_run = st.button("Continuation çalıştır", key="run_cont")

    cont_use_full_market = st.checkbox("NASDAQ + NYSE tam evreni tara", value=False, key="cont_use_full_market")
    if cont_use_full_market:
        st.info("İlk tam evren taraması yavaş olabilir. Başlangıç için Evren=1500, Ön filtre=120, Detaylı scan=50 önerilir.")
    cfm1, cfm2, cfm3, cfm4 = st.columns(4)
    with cfm1:
        cont_exchanges = st.multiselect("Borsalar", ["NASDAQ", "NYSE"], default=["NASDAQ", "NYSE"], key="cont_exchanges")
    with cfm2:
        cont_prefilter_limit = st.slider("Ön filtre aday sayısı", 50, 500, 120, 10, key="cont_prefilter_limit")
    with cfm3:
        cont_deep_scan_limit = st.slider("Detaylı scan aday sayısı", 20, 250, 50, 10, key="cont_deep_scan_limit")
    with cfm4:
        cont_universe_limit = st.number_input("Evren sınırı (0=tamı / yavaş)", min_value=0, value=1500, step=500, key="cont_universe_limit")

    if cont_run:
        if cont_use_full_market and not have_alpaca():
            st.warning("Tam evren taraması için Alpaca API gerekir.")
        else:
            trade_date_for_pref = ny_date_str() if cont_mode == "Canlı" else str(cont_date)
            symbols, cont_meta, cont_pre_df = resolve_scan_symbols(
                cont_symbols_text, cont_use_full_market, "continuation", trade_date_for_pref,
                cont_exchanges, cont_prefilter_limit, cont_deep_scan_limit, cont_universe_limit
            )

            if not symbols:
                st.warning("Sembol bulunamadı.")
            else:
                with st.spinner("Continuation Engine çalışıyor..."):
                    if cont_mode == "Canlı":
                        df, session, cutoff = build_engine_rows(
                            symbols, "continuation",
                            cont_score_al=CONT_SCORE_AL, cont_score_strong=CONT_SCORE_STRONG,
                            super_pattern_al=SUPER_PATTERN_AL, super_pattern_strong=SUPER_PATTERN_STRONG
                        )
                    else:
                        df, session, cutoff = build_engine_rows(
                            symbols, "continuation",
                            trade_date_str=str(cont_date), cutoff_time="09:25:00",
                            cont_score_al=CONT_SCORE_AL, cont_score_strong=CONT_SCORE_STRONG,
                            super_pattern_al=SUPER_PATTERN_AL, super_pattern_strong=SUPER_PATTERN_STRONG
                        )

                st.write(f"**Session:** {session} | **Cutoff:** {cutoff or '-'}")
                render_session_message(session)
                if cont_use_full_market:
                    st.info(f"Evren: {cont_meta['universe_count']} | ranked: {cont_meta['ranked_count']} | deep scan: {cont_meta['deep_count']}")

                if df.empty:
                    st.warning("Sonuç yok.")
                else:
                    if SHOW_ONLY_TRADEABLE:
                        df = df[df["Decision"].isin(["GÜÇLÜ AL", "AL", "İZLE"])].copy()

                    tops = top_cards(df, 3)
                    cols = st.columns(3)
                    for i, item in enumerate(tops):
                        with cols[i]:
                            st.metric(
                                f"#{i+1} {item['Symbol']} — {item['Decision']}",
                                f"Score {item['Score']}",
                                f"Entry {format_price(item['Entry_Idea']) if pd.notna(item['Entry_Idea']) else '-'}"
                            )
                            st.caption(item["Notes"])

                    st.dataframe(df.head(N_SHOW), use_container_width=True)
                    csv = df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 Continuation CSV indir",
                        data=csv,
                        file_name=f"continuation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_cont"
                    )

                    if cont_use_full_market:
                        with st.expander("Continuation ön filtre adayları"):
                            st.dataframe(cont_pre_df.head(min(N_SHOW, len(cont_pre_df))), use_container_width=True)

with tab2:
    st.subheader("Supernova Engine")
    st.caption("Amaç: RMSG / STI tipi düşük fiyatlı, patlayıcı, erken ignition adaylarını bulmak.")
    col1, col2 = st.columns([2, 1])

    with col1:
        rocket_symbols_text = st.text_area(
            "Sembol listesi",
            value=",".join(DEFAULT_SUPERNOVA_SYMBOLS),
            height=100,
            key="rocket_symbols"
        )
    with col2:
        rocket_mode = st.radio("Mod", ["Canlı", "Backtest"], index=0, key="rocket_mode")
        rocket_date = st.date_input("Backtest tarihi", value=now_ny().date(), key="rocket_date")
        rocket_run = st.button("Supernova çalıştır", key="run_rocket")

    rocket_use_full_market = st.checkbox("NASDAQ + NYSE tam evreni tara", value=False, key="rocket_use_full_market")
    if rocket_use_full_market:
        st.info("İlk tam evren taraması yavaş olabilir. Başlangıç için Evren=1500, Ön filtre=120, Detaylı scan=50 önerilir.")
    sfm1, sfm2, sfm3, sfm4 = st.columns(4)
    with sfm1:
        rocket_exchanges = st.multiselect("Borsalar", ["NASDAQ", "NYSE"], default=["NASDAQ", "NYSE"], key="rocket_exchanges")
    with sfm2:
        rocket_prefilter_limit = st.slider("Ön filtre aday sayısı", 50, 500, 120, 10, key="rocket_prefilter_limit")
    with sfm3:
        rocket_deep_scan_limit = st.slider("Detaylı scan aday sayısı", 20, 250, 50, 10, key="rocket_deep_scan_limit")
    with sfm4:
        rocket_universe_limit = st.number_input("Evren sınırı (0=tamı / yavaş)", min_value=0, value=1500, step=500, key="rocket_universe_limit")

    if rocket_run:
        if rocket_use_full_market and not have_alpaca():
            st.warning("Tam evren taraması için Alpaca API gerekir.")
        else:
            trade_date_for_pref = ny_date_str() if rocket_mode == "Canlı" else str(rocket_date)
            symbols, super_meta, super_pre_df = resolve_scan_symbols(
                rocket_symbols_text, rocket_use_full_market, "supernova", trade_date_for_pref,
                rocket_exchanges, rocket_prefilter_limit, rocket_deep_scan_limit, rocket_universe_limit
            )

            if not symbols:
                st.warning("Sembol bulunamadı.")
            else:
                with st.spinner("Supernova Engine çalışıyor..."):
                    if rocket_mode == "Canlı":
                        df, session, cutoff = build_engine_rows(
                            symbols, "supernova",
                            cont_score_al=CONT_SCORE_AL, cont_score_strong=CONT_SCORE_STRONG,
                            super_pattern_al=SUPER_PATTERN_AL, super_pattern_strong=SUPER_PATTERN_STRONG
                        )
                    else:
                        df, session, cutoff = build_engine_rows(
                            symbols, "supernova",
                            trade_date_str=str(rocket_date), cutoff_time="09:25:00",
                            cont_score_al=CONT_SCORE_AL, cont_score_strong=CONT_SCORE_STRONG,
                            super_pattern_al=SUPER_PATTERN_AL, super_pattern_strong=SUPER_PATTERN_STRONG
                        )

                st.write(f"**Session:** {session} | **Cutoff:** {cutoff or '-'}")
                render_session_message(session)
                if rocket_use_full_market:
                    st.info(f"Evren: {super_meta['universe_count']} | ranked: {super_meta['ranked_count']} | deep scan: {super_meta['deep_count']}")

                if df.empty:
                    st.warning("Sonuç yok.")
                else:
                    if SHOW_ONLY_TRADEABLE:
                        df = df[df["Decision"].isin(["GÜÇLÜ AL", "AL", "İZLE"])].copy()

                    tops = top_cards(df, 3)
                    cols = st.columns(3)
                    for i, item in enumerate(tops):
                        with cols[i]:
                            st.metric(
                                f"#{i+1} {item['Symbol']} — {item['Decision']}",
                                f"Pattern {item['Pattern_Score']}",
                                f"Entry {format_price(item['Entry_Idea']) if pd.notna(item['Entry_Idea']) else '-'}"
                            )
                            st.caption(f"{item['Best_Pattern']} | {item['Notes']}")

                    st.dataframe(df.head(N_SHOW), use_container_width=True)
                    csv = df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 Supernova CSV indir",
                        data=csv,
                        file_name=f"supernova_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_supernova"
                    )

                    if rocket_use_full_market:
                        with st.expander("Supernova ön filtre adayları"):
                            st.dataframe(super_pre_df.head(min(N_SHOW, len(super_pre_df))), use_container_width=True)

with tab3:
    st.subheader("ML Trainer")
    st.caption("Kural tabanlı motorun üstüne olasılık katmanı eklemek için supervision tabanlı model eğitir.")

    ml_symbols_text = st.text_area(
        "Eğitim sembol listesi",
        value=",".join(DEFAULT_ML_TRAIN_SYMBOLS),
        height=120,
        key="ml_symbols"
    )
    c1, c2 = st.columns(2)
    with c1:
        train_cont_btn = st.button("Continuation model eğit", key="train_cont_model")
    with c2:
        train_super_btn = st.button("Supernova model eğit", key="train_super_model")

    st.markdown("### Kayıtlı modeller")
    cont_model, cont_cols, cont_meta = load_model_bundle("continuation_model")
    super_model, super_cols, super_meta = load_model_bundle("supernova_model")

    meta_df = pd.DataFrame([
        {
            "Model": "continuation_model",
            "Loaded": cont_model is not None,
            "Rows": cont_meta.get("rows") if cont_meta else None,
            "Train_Rows": cont_meta.get("train_rows") if cont_meta else None,
            "Test_Rows": cont_meta.get("test_rows") if cont_meta else None,
            "Test_Acc@0.5": cont_meta.get("test", {}).get("accuracy_0_5") if cont_meta else None,
            "Test_Precision@0.5": cont_meta.get("test", {}).get("precision_0_5") if cont_meta else None,
            "Test_Recall@0.5": cont_meta.get("test", {}).get("recall_0_5") if cont_meta else None,
        },
        {
            "Model": "supernova_model",
            "Loaded": super_model is not None,
            "Rows": super_meta.get("rows") if super_meta else None,
            "Train_Rows": super_meta.get("train_rows") if super_meta else None,
            "Test_Rows": super_meta.get("test_rows") if super_meta else None,
            "Test_Acc@0.5": super_meta.get("test", {}).get("accuracy_0_5") if super_meta else None,
            "Test_Precision@0.5": super_meta.get("test", {}).get("precision_0_5") if super_meta else None,
            "Test_Recall@0.5": super_meta.get("test", {}).get("recall_0_5") if super_meta else None,
        },
    ])
    st.dataframe(meta_df, use_container_width=True)

    if train_cont_btn or train_super_btn:
        syms = parse_symbols(ml_symbols_text)
        if not syms:
            st.warning("Eğitim için sembol gir.")
        else:
            model_type = "continuation" if train_cont_btn else "supernova"
            with st.spinner(f"{model_type} dataset hazırlanıyor..."):
                ds = build_training_dataset(syms, model_type)

            if ds.empty:
                st.error("Dataset oluşmadı.")
            else:
                st.write(f"Dataset rows: {len(ds)}")
                st.dataframe(ds.head(20), use_container_width=True)

                with st.spinner(f"{model_type} model eğitiliyor..."):
                    model, feature_cols, metrics = train_model_from_dataset(ds, model_type)

                model_name = f"{model_type}_model"
                save_model_bundle(model_name, model, feature_cols, metrics)
                st.success(f"{model_name} kaydedildi.")
                st.json(metrics)

                csv = ds.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    f"📥 {model_type} dataset CSV indir",
                    data=csv,
                    file_name=f"{model_type}_training_dataset.csv",
                    mime="text/csv",
                    key=f"download_{model_type}_dataset"
                )

with tab4:
    st.subheader("Risk Monitor")
    st.caption("Actionable sinyallerin execution/risk filtresi. Piyasa kapalıysa sonuçlar watchlist-only olarak işaretlenir.")

    risk_symbols_text = st.text_area(
        "Risk monitor sembol listesi",
        value=",".join(list(dict.fromkeys(DEFAULT_CONTINUATION_SYMBOLS[:8] + DEFAULT_SUPERNOVA_SYMBOLS[:8]))),
        height=100,
        key="risk_symbols"
    )
    risk_run = st.button("Risk monitor çalıştır", key="risk_run")

    if risk_run:
        syms = parse_symbols(risk_symbols_text)
        if not syms:
            st.warning("Sembol gir.")
        else:
            with st.spinner("Risk monitor çalışıyor..."):
                cont_df, _, _ = build_engine_rows(
                    syms, "continuation",
                    cont_score_al=CONT_SCORE_AL, cont_score_strong=CONT_SCORE_STRONG,
                    super_pattern_al=SUPER_PATTERN_AL, super_pattern_strong=SUPER_PATTERN_STRONG
                )
                super_df, _, _ = build_engine_rows(
                    syms, "supernova",
                    cont_score_al=CONT_SCORE_AL, cont_score_strong=CONT_SCORE_STRONG,
                    super_pattern_al=SUPER_PATTERN_AL, super_pattern_strong=SUPER_PATTERN_STRONG
                )

            frames = []
            if not cont_df.empty:
                frames.append(cont_df[[
                    "Engine", "Symbol", "Decision", "ML_Prob", "Entry_Idea", "Stop", "TP1", "TP2",
                    "Late_Status", "Extension_%", "Bid", "Ask", "Spread_%", "Stop_Dist_%",
                    "Risk_Grade", "Real_Money_Allowed", "Risk_Tags"
                ]].copy())
            if not super_df.empty:
                frames.append(super_df[[
                    "Engine", "Symbol", "Decision", "ML_Prob", "Entry_Idea", "Stop", "TP1", "TP2",
                    "Late_Status", "Extension_%", "Bid", "Ask", "Spread_%", "Stop_Dist_%",
                    "Risk_Grade", "Real_Money_Allowed", "Risk_Tags"
                ]].copy())

            if not frames:
                st.warning("Risk sonucu yok.")
            else:
                risk_df = pd.concat(frames, ignore_index=True)
                risk_df["_rank"] = risk_df["Real_Money_Allowed"].map({True: 0, False: 1})
                risk_df = risk_df.sort_values(["_rank", "Risk_Grade", "Decision"]).drop(columns=["_rank"])
                st.dataframe(risk_df.head(N_SHOW), use_container_width=True)
                csv = risk_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 Risk CSV indir",
                    data=csv,
                    file_name=f"risk_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_risk"
                )

with tab5:
    render_journal_tab()


with tab6:
    st.subheader("Overnight Engine")
    st.caption("Amaç: gece alıp ertesi gün premarkette veya regular session'da satılabilecek continuation adaylarını bulmak.")

    col1, col2 = st.columns([2, 1])
    with col1:
        ov_symbols_text = st.text_area(
            "Sembol listesi",
            value=",".join(DEFAULT_CONTINUATION_SYMBOLS),
            height=100,
            key="ov_symbols"
        )
    with col2:
        ov_mode = st.radio("Mod", ["Canlı", "Backtest"], index=0, key="ov_mode")
        ov_date = st.date_input("Backtest exit günü", value=now_ny().date(), key="ov_date")
        ov_run = st.button("Overnight çalıştır", key="run_overnight")

    ov_use_full_market = st.checkbox("NASDAQ + NYSE tam evreni tara", value=False, key="ov_use_full_market")
    if ov_use_full_market:
        st.info("Başlangıç için Evren=1500, Ön filtre=120, Detaylı scan=50 önerilir.")
    ofm1, ofm2, ofm3, ofm4 = st.columns(4)
    with ofm1:
        ov_exchanges = st.multiselect("Borsalar", ["NASDAQ", "NYSE"], default=["NASDAQ", "NYSE"], key="ov_exchanges")
    with ofm2:
        ov_prefilter_limit = st.slider("Ön filtre aday sayısı", 50, 500, 120, 10, key="ov_prefilter_limit")
    with ofm3:
        ov_deep_scan_limit = st.slider("Detaylı scan aday sayısı", 20, 250, 50, 10, key="ov_deep_scan_limit")
    with ofm4:
        ov_universe_limit = st.number_input("Evren sınırı (0=tamı / yavaş)", min_value=0, value=1500, step=500, key="ov_universe_limit")

    if ov_run:
        if ov_use_full_market and not have_alpaca():
            st.warning("Tam evren taraması için Alpaca API gerekir.")
        else:
            trade_date_for_pref = overnight_reference_date_str(get_session_label()) if ov_mode == "Canlı" else str(ov_date)
            symbols, ov_meta, ov_pre_df = resolve_scan_symbols(
                ov_symbols_text, ov_use_full_market, "overnight", trade_date_for_pref,
                ov_exchanges, ov_prefilter_limit, ov_deep_scan_limit, ov_universe_limit
            )

            if not symbols:
                st.warning("Sembol bulunamadı.")
            else:
                with st.spinner("Overnight Engine çalışıyor..."):
                    df, session, ref_date = build_overnight_rows(symbols, None if ov_mode == "Canlı" else str(ov_date))

                st.write(f"**Session:** {session} | **Reference Exit Date:** {ref_date}")
                if ov_use_full_market:
                    st.caption(
                        f"Source={ov_meta['source']} | Universe={ov_meta['universe_count']} | Ranked={ov_meta['ranked_count']} | Deep={ov_meta['deep_count']}"
                    )

                if df.empty:
                    st.warning("Sonuç yok.")
                else:
                    if SHOW_ONLY_TRADEABLE:
                        df = df[df["Decision"].isin(["GÜÇLÜ GECE AL", "GECE AL", "İZLE"])].copy()

                    tops = top_cards(df, 3)
                    cols = st.columns(3)
                    for i, item in enumerate(tops):
                        with cols[i]:
                            st.metric(
                                f"#{i+1} {item['Symbol']} — {item['Decision']}",
                                f"Night {item['Night_Score']}",
                                f"Entry {format_price(item['Entry_Idea']) if pd.notna(item['Entry_Idea']) else '-'}"
                            )
                            st.caption(f"{item['Trade_Phase']} | {item['Action_Status']} | {item['Notes']}")

                    st.dataframe(df.head(N_SHOW), use_container_width=True)
                    if ov_use_full_market and not ov_pre_df.empty:
                        with st.expander("Ön filtre adayları"):
                            st.dataframe(ov_pre_df.head(min(N_SHOW, len(ov_pre_df))), use_container_width=True)

                    csv = df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 Overnight CSV indir",
                        data=csv,
                        file_name=f"overnight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_overnight"
                    )

with tab7:
    st.subheader("Premarket Ignition")
    st.caption("Amaç: premarkette ateşleyen, erken giriş verip henüz çok uzamamış adayları bulmak.")

    col1, col2 = st.columns([2, 1])
    with col1:
        ign_symbols_text = st.text_area(
            "Sembol listesi",
            value=",".join(DEFAULT_SUPERNOVA_SYMBOLS),
            height=100,
            key="ign_symbols"
        )
    with col2:
        ign_mode = st.radio("Mod", ["Canlı", "Backtest"], index=0, key="ign_mode")
        ign_date = st.date_input("Backtest günü", value=now_ny().date(), key="ign_date")
        ign_run = st.button("Premarket ignition çalıştır", key="run_ignition")

    ign_use_full_market = st.checkbox("NASDAQ + NYSE tam evreni tara", value=False, key="ign_use_full_market")
    if ign_use_full_market:
        st.info("Başlangıç için Evren=1500, Ön filtre=120, Detaylı scan=50 önerilir.")
    ifm1, ifm2, ifm3, ifm4 = st.columns(4)
    with ifm1:
        ign_exchanges = st.multiselect("Borsalar", ["NASDAQ", "NYSE"], default=["NASDAQ", "NYSE"], key="ign_exchanges")
    with ifm2:
        ign_prefilter_limit = st.slider("Ön filtre aday sayısı", 50, 500, 120, 10, key="ign_prefilter_limit")
    with ifm3:
        ign_deep_scan_limit = st.slider("Detaylı scan aday sayısı", 20, 250, 50, 10, key="ign_deep_scan_limit")
    with ifm4:
        ign_universe_limit = st.number_input("Evren sınırı (0=tamı / yavaş)", min_value=0, value=1500, step=500, key="ign_universe_limit")

    if ign_run:
        if ign_use_full_market and not have_alpaca():
            st.warning("Tam evren taraması için Alpaca API gerekir.")
        else:
            trade_date_for_pref = ny_date_str() if ign_mode == "Canlı" else str(ign_date)
            symbols, ign_meta, ign_pre_df = resolve_scan_symbols(
                ign_symbols_text, ign_use_full_market, "premarket_ignition", trade_date_for_pref,
                ign_exchanges, ign_prefilter_limit, ign_deep_scan_limit, ign_universe_limit
            )

            if not symbols:
                st.warning("Sembol bulunamadı.")
            else:
                with st.spinner("Premarket Ignition çalışıyor..."):
                    df, session, ref_date = build_premarket_ignition_rows(symbols, None if ign_mode == "Canlı" else str(ign_date))

                st.write(f"**Session:** {session} | **Reference Date:** {ref_date}")
                if ign_use_full_market:
                    st.caption(
                        f"Source={ign_meta['source']} | Universe={ign_meta['universe_count']} | Ranked={ign_meta['ranked_count']} | Deep={ign_meta['deep_count']}"
                    )

                if df.empty:
                    st.warning("Sonuç yok.")
                else:
                    if SHOW_ONLY_TRADEABLE:
                        df = df[df["Decision"].isin(["PREMARKET GÜÇLÜ AL", "PREMARKET AL", "İZLE", "GEÇ KALINDI"])].copy()

                    tops = top_cards(df, 3)
                    cols = st.columns(3)
                    for i, item in enumerate(tops):
                        with cols[i]:
                            st.metric(
                                f"#{i+1} {item['Symbol']} — {item['Decision']}",
                                f"Ignition {item['Ignition_Score']}",
                                f"Entry {format_price(item['Entry_Idea']) if pd.notna(item['Entry_Idea']) else '-'}"
                            )
                            st.caption(f"{item['Trade_Phase']} | {item['Action_Status']} | {item['Notes']}")

                    st.dataframe(df.head(N_SHOW), use_container_width=True)
                    if ign_use_full_market and not ign_pre_df.empty:
                        with st.expander("Ön filtre adayları"):
                            st.dataframe(ign_pre_df.head(min(N_SHOW, len(ign_pre_df))), use_container_width=True)

                    csv = df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 Premarket ignition CSV indir",
                        data=csv,
                        file_name=f"premarket_ignition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_ignition"
                    )
