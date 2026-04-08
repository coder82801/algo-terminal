import os
import streamlit as st
import alpaca_trade_api as tradeapi
import pandas as pd
import requests
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Algo-Trading Terminal", layout="wide")
st.title("🎯 Hibrit Momentum & İşlem Terminali (v5.0 - Çoklu Oturum)")

# Render'ın güvenli kasasından şifreleri çekiyoruz
env_api_key = os.getenv("ALPACA_API_KEY", "")
env_secret_key = os.getenv("ALPACA_SECRET_KEY", "")

# --- YAN MENÜ: API GİRİŞİ ---
st.sidebar.header("Alpaca API (Paper)")
api_key = st.sidebar.text_input("API Key ID", value=env_api_key, type="password")
secret_key = st.sidebar.text_input("Secret Key", value=env_secret_key, type="password")

def get_api(api_key, secret_key):
    return tradeapi.REST(
        key_id=api_key, 
        secret_key=secret_key, 
        base_url='https://paper-api.alpaca.markets', 
        api_version='v2'
    )

# --- MODÜL 1: CANLI MARKET MOVERS ---
st.subheader("1. Aşama: Canlı Piyasa Tarayıcı")

# --- SEANS SEÇİM EKRANI ---
session_choice = st.radio(
    "Piyasa Oturumunu Seçin (Market Session):",
    ["☀️ Gün İçi (Intraday)", "🌅 Piyasa Öncesi (Pre-Market)", "🌙 Kapanış Sonrası (After-Hours)"],
    horizontal=True
)

col_btn, col_chk = st.columns([1, 3])
with col_btn:
    refresh_btn = st.button("🔄 Manuel Yenile")
with col_chk:
    auto_refresh = st.checkbox("⚡ 15 Saniyede Bir Otomatik Yenile (Canlı Mod)", value=True)

if auto_refresh:
    st_autorefresh(interval=15000, key="auto_refresh_gainers")

@st.cache_data(ttl=10)
def get_top_gainers(session):
    try:
        # Seans tipine göre TradingView filtrelerini dinamik olarak ayarlıyoruz
        if "Pre-Market" in session:
            sort_field = "premarket_change"
            vol_field = "premarket_volume"
            price_field = "premarket_close"
            min_vol = 10000 # Pre-market sığ olduğu için 10k hacim yeterli
        elif "After-Hours" in session:
            sort_field = "postmarket_change"
            vol_field = "postmarket_volume"
            price_field = "postmarket_close"
            min_vol = 10000
        else: # Intraday (Gün İçi)
            sort_field = "change"
            vol_field = "volume"
            price_field = "close"
            min_vol = 50000 # Gün içi hacim filtresi 50k
            
        url_tv = "https://scanner.tradingview.com/america/scan"
        payload = {
            "filter": [
                {"left": vol_field, "operation": "greater", "right": min_vol}, 
                {"left": price_field, "operation": "greater", "right": 0.50}, 
                {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]} 
            ],
            "options": {"lang": "en"},
            "markets": ["america"],
            "symbols": {"query": {"types": ["stock"]}, "tickers": []},
            "columns": ["name", "description", price_field, sort_field, vol_field],
            "sort": {"sortBy": sort_field, "sortOrder": "desc"},
            "range": [0, 20] 
        }
        headers_tv = {"User-Agent": "Mozilla/5.0"}
        
        res_tv = requests.post(url_tv, json=payload, headers=headers_tv, timeout=10)
        res_tv.raise_for_status() 
        tv_data = res_tv.json()

        tickers = [item['d'][0] for item in tv_data.get('data', [])]
        if not tickers:
            return pd.DataFrame()

        # Yahoo Finance ile anlık canlı fiyatları çekiyoruz (Pre/Post market dahil)
        try:
            yf_data = yf.download(tickers, period="1d", interval="1m", prepost=True, progress=False)
            closes = yf_data['Close']
        except Exception:
            closes = pd.DataFrame()

        results = []
        for item in tv_data.get('data', []):
            ticker = item['d'][0]
            sirket = item['d'][1]
            
            # API'den bazen "null" gelebilir, bu durumu önlüyoruz
            tv_fiyat = item['d'][2] if item['d'][2] is not None else 0
            tv_degisim = item['d'][3] if item['d'][3] is not None else 0
            tv_hacim = item['d'][4] if item['d'][4] is not None else 0
            
            # Mümkünse %100 canlı Yahoo verisini kullan, yoksa TV verisini kullan
            try:
                if not closes.empty:
                    if len(tickers) > 1:
                        fiyat = float(closes[ticker].ffill().iloc[-1])
                    else:
                        fiyat = float(closes.ffill().iloc[-1])
                else:
                    fiyat = tv_fiyat
            except Exception:
                fiyat = tv_fiyat
                
            fiyat_format = round(fiyat, 4) if fiyat < 1 else round(fiyat, 2)
            
            results.append({
                'Hisse': ticker,
                'Şirket': sirket,
                'Son Fiyat ($)': fiyat_format,
                f'{session.split(" ")[1]} Artışı (%)': round(tv_degisim, 2),
                'Hacim': f"{int(tv_hacim):,}"
            })
            
        df = pd.DataFrame(results)
        df = df.sort_values(by=df.columns[3], ascending=False).reset_index(drop=True)
        return df

    except Exception as e:
        return pd.DataFrame()

# Tablonun Ekrana Yazdırılması (Seçilen seansa göre güncellenir)
with st.spinner(f"{session_choice} verileri çekiliyor..."):
    df_gainers = get_top_gainers(session_choice)
    if not df_gainers.empty:
        st.dataframe(df_gainers, use_container_width=True)
    else:
        st.warning(f"{session_choice} oturumu için şu an hareketli hisse bulunamadı veya piyasa kapalı.")

st.divider()

# --- MODÜL 2: TEKNİK ANALİZ (VWAP) ---
st.subheader("2. Aşama: Akıllı Analiz ve İşlem")

ticker = st.text_input("İşlem Yapılacak Hisse Sembolü (Örn: CAR)", "").upper()

if ticker:
    if st.button(f"🔍 {ticker} İçin VWAP Analizi Yap"):
        with st.spinner("Grafikler inceleniyor ve seviyeler hesaplanıyor..."):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period='1d', interval='5m', prepost=True)
                
                if not df.empty:
                    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
                    df['VP'] = df['Typical_Price'] * df['Volume']
                    df['VWAP'] = df['VP'].cumsum() / df['Volume'].cumsum()
                    
                    current_price = round(df['Close'].iloc[-1], 2)
                    vwap_price = round(df['VWAP'].iloc[-1], 2)
                    day_high = round(df['High'].max(), 2)
                    
                    st.info("### 📊 Analiz Raporu")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Güncel Fiyat (Pre/Post Market Dahil)", f"${current_price}")
                    col_b.metric("Gün İçi Zirve", f"${day_high}")
                    col_c.metric("VWAP (Referans)", f"${vwap_price}")
                    
                    st.success(f"**Önerilen Strateji:** Fiyatın **${vwap_price}** seviyesindeki VWAP desteğine çekilmesini bekleyin. Limit alış emrinizi bu seviyeye yakın kurun.")
                else:
                    st.warning("Bu hisse için gün içi grafik verisi bulunamadı.")
            except Exception as e:
                st.error(f"Analiz sırasında bir hata oluştu: {e}")

st.write("---")

# --- MODÜL 3: ALPACA EMİR GÖNDERİMİ ---
if api_key and secret_key:
    try:
        api = get_api(api_key, secret_key)
        
        with st.form("bracket_order_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                qty = st.number_input("Adet", min_value=1, value=100)
                limit_price = st.number_input("Alış Limit Fiyatı ($) - VWAP'a yakın tutun", min_value=0.01, value=1.00, step=0.01)
                
            with col2:
                take_profit_price = st.number_input("Kar-Al Fiyatı ($) -> %15 Hedef", min_value=0.01, value=limit_price * 1.15, step=0.01)
                stop_loss_price = st.number_input("Zarar-Kes Fiyatı ($) -> %5 Risk", min_value=0.01, value=limit_price * 0.95, step=0.01)

            submit_button = st.form_submit_button("🚀 Emri Piyasaya Gönder")

            if submit_button and ticker:
                try:
                    api.submit_order(
                        symbol=ticker, qty=qty, side='buy', type='limit', time_in_force='day',
                        limit_price=limit_price, extended_hours=True, order_class='bracket',
                        take_profit=dict(limit_price=round(take_profit_price, 2)),
                        stop_loss=dict(stop_price=round(stop_loss_price, 2), limit_price=round(stop_loss_price - 0.02, 2))
                    )
                    st.success(f"İşlem Başarılı! {ticker} için {limit_price}$ seviyesinden emir iletildi.")
                    st.info(f"Hedef: {round(take_profit_price, 2)}$ | Stop: {round(stop_loss_price, 2)}$")
                except Exception as e:
                    st.error(f"Emir Hatası: {e}")
    except Exception:
        pass
