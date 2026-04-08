import os
import streamlit as st
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Algo-Trading Terminal", layout="wide")
st.title("🎯 Hibrit Momentum & İşlem Terminali (v6.1)")

env_api_key = os.getenv("ALPACA_API_KEY", "")
env_secret_key = os.getenv("ALPACA_SECRET_KEY", "")

st.sidebar.header("Alpaca API (Paper)")
api_key = st.sidebar.text_input("API Key ID", value=env_api_key, type="password")
secret_key = st.sidebar.text_input("Secret Key", value=env_secret_key, type="password")

def get_api(api_key, secret_key):
    return tradeapi.REST(key_id=api_key, secret_key=secret_key, base_url='https://paper-api.alpaca.markets', api_version='v2')

# --- SEKME OLUŞTURMA ---
tab1, tab2 = st.tabs(["⚡ Canlı Gün İçi Radar", "🔮 Kurumsal Swing Radar (Ertesi Gün)"])

# ==========================================
# SEKME 1: CANLI GÜN İÇİ RADAR
# ==========================================
with tab1:
    session_choice = st.radio(
        "Piyasa Oturumunu Seçin (Market Session):",
        ["☀️ Gün İçi (Intraday)", "🌅 Piyasa Öncesi (Pre-Market)", "🌙 Kapanış Sonrası (After-Hours)"],
        horizontal=True
    )

    col_btn, col_chk = st.columns([1, 3])
    with col_btn:
        refresh_btn = st.button("🔄 Manuel Yenile", key="btn_refresh_1")
    with col_chk:
        auto_refresh = st.checkbox("⚡ 15 Saniyede Bir Otomatik Yenile", value=True)

    if auto_refresh:
        st_autorefresh(interval=15000, key="auto_refresh_gainers")

    @st.cache_data(ttl=10)
    def get_intraday_gainers(session):
        try:
            if "Pre-Market" in session:
                sort_field, vol_field, price_field, min_vol = "premarket_change", "premarket_volume", "premarket_close", 10000
            elif "After-Hours" in session:
                sort_field, vol_field, price_field, min_vol = "postmarket_change", "postmarket_volume", "postmarket_close", 10000
            else: 
                sort_field, vol_field, price_field, min_vol = "change", "volume", "close", 50000
                
            url = "https://scanner.tradingview.com/america/scan"
            payload = {
                "filter": [
                    {"left": vol_field, "operation": "greater", "right": min_vol}, 
                    {"left": price_field, "operation": "greater", "right": 0.50}, 
                    {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]}
                ],
                "options": {"lang": "en"}, "markets": ["america"],
                "symbols": {"query": {"types": ["stock"]}, "tickers": []},
                "columns": ["name", "description", price_field, sort_field, vol_field],
                "sort": {"sortBy": sort_field, "sortOrder": "desc"},
                "range": [0, 15] 
            }
            res = requests.post(url, json=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            data = res.json()
            
            results = []
            for item in data.get('data', []):
                d = item['d']
                fiyat = round(d[2], 4) if d[2] and d[2] < 1 else round(d[2] or 0, 2)
                results.append({'Hisse': d[0], 'Şirket': d[1], 'Fiyat ($)': fiyat, 'Artış (%)': round(d[3] or 0, 2), 'Hacim': f"{int(d[4] or 0):,}"})
            return pd.DataFrame(results)
        except: return pd.DataFrame()

    df_gainers = get_intraday_gainers(session_choice)
    if not df_gainers.empty:
        st.dataframe(df_gainers, use_container_width=True)
    else:
        st.info("Veri bekleniyor...")

# ==========================================
# SEKME 2: KURUMSAL SWING RADAR (GÜÇLENDİRİLMİŞ PYTHON MOTORU)
# ==========================================
with tab2:
    st.write("Profesyonel filtrelere göre 'Ertesi Gün' (Swing) potansiyeli yüksek adaylar:")
    
    algo_choice = st.selectbox(
        "Kullanılacak Algoritmayı Seçin:",
        [
            "A) Ertesi Gün Breakout (Fiyat>20EMA>50EMA, Top %30 Kapanış, RVOL>1.5, Sıkışma)",
            "B) İkinci Gün Koşusu (Gap-Up, RVOL>3, VWAP Üstü Kapanış)",
            "C) Kurumsal Birikim (200MA Üstü, Hacimli Alışlar, Sıkışma)"
        ]
    )
    
    if st.button("🚀 Kurumsal Taramayı Başlat (Zaman Alabilir)"):
        with st.spinner("1. Aşama: Piyasadan hacimli hisseler toplanıyor..."):
            try:
                # Sadece hacimli ve fiyatı uygun 80 hisseyi al (Filtreyi Python yapacak)
                url = "https://scanner.tradingview.com/america/scan"
                payload = {
                    "filter": [
                        {"left": "close", "operation": "greater", "right": 2.00},
                        {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
                        {"left": "volume", "operation": "greater", "right": 500000}
                    ],
                    "options": {"lang": "en"}, "markets": ["america"],
                    "symbols": {"query": {"types": ["stock"]}, "tickers": []},
                    "columns": ["name"], "sort": {"sortBy": "change", "sortOrder": "desc"},
                    "range": [0, 80] 
                }
                res = requests.post(url, json=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                tickers = [item['d'][0] for item in res.json().get('data', [])]
                
                if not tickers:
                    st.warning("Hisse çekilemedi.")
                else:
                    st.write(f"2. Aşama: {len(tickers)} hisse üzerinde algoritmik matematik testi yapılıyor...")
                    yf_data = yf.download(tickers, period="60d", progress=False)
                    
                    final_candidates = []
                    
                    for ticker in tickers:
                        try:
                            if ticker not in yf_data['Close'].columns: continue
                            
                            df = pd.DataFrame({
                                'Close': yf_data['Close'][ticker],
                                'High': yf_data['High'][ticker],
                                'Low': yf_data['Low'][ticker],
                                'Volume': yf_data['Volume'][ticker],
                                'Open': yf_data['Open'][ticker]
                            }).dropna()
                            
                            if len(df) < 50: continue
                            
                            # Gösterge Hesaplamaları (Tümü Python'da yapılıyor)
                            ema20 = df['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
                            ema50 = df['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
                            sma200 = df['Close'].rolling(window=50).mean().iloc[-1] # 60 günlük veride 50MA'yı proxy olarak kullanıyoruz
                            
                            last_close = df['Close'].iloc[-1]
                            last_high = df['High'].iloc[-1]
                            last_low = df['Low'].iloc[-1]
                            last_vol = df['Volume'].iloc[-1]
                            
                            avg_vol_10d = df['Volume'].rolling(10).mean().iloc[-2] # Bugünü hariç tutarak son 10 gün ortalaması
                            rvol = last_vol / avg_vol_10d if avg_vol_10d > 0 else 0
                            
                            daily_range = last_high - last_low
                            close_position = (last_close - last_low) / daily_range if daily_range > 0 else 0
                            
                            df['TR'] = df[['High', 'Low', 'Close']].max(axis=1) - df[['High', 'Low', 'Close']].min(axis=1)
                            atr5 = df['TR'].rolling(5).mean().iloc[-1]
                            atr20 = df['TR'].rolling(20).mean().iloc[-1]
                            
                            if "A)" in algo_choice:
                                # TESTLER
                                cond1 = last_close > ema20 > ema50  # Trend şartı
                                cond2 = rvol > 1.5                  # Hacim şartı (2.5 çok nadirdir, 1.5'e çektik)
                                cond3 = close_position >= 0.70      # Zirveye yakın kapanış (Top %30)
                                cond4 = atr5 <= atr20 * 1.2         # Daralma şartı (Patlama gününde ATR hafif artabileceği için esnetildi)
                                
                                if cond1 and cond2 and cond3 and cond4:
                                    final_candidates.append({
                                        'Hisse': ticker, 
                                        'Puan/Durum': 'Breakout Adayı (RVOL, Top %30 Kap.)', 
                                        'Fiyat': round(last_close, 2),
                                        'RVOL': round(rvol, 2)
                                    })
                            
                            elif "B)" in algo_choice:
                                gap = (df['Open'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
                                if gap > 2.0 and rvol > 3.0 and close_position >= 0.70:
                                    final_candidates.append({'Hisse': ticker, 'Puan/Durum': 'Gap-Up ve İkinci Gün Potansiyeli', 'Fiyat': round(last_close, 2), 'RVOL': round(rvol, 2)})
                                    
                            elif "C)" in algo_choice:
                                up_vol = df[df['Close'] > df['Open']]['Volume'].tail(15).mean()
                                down_vol = df[df['Close'] < df['Open']]['Volume'].tail(15).mean()
                                if last_close > sma200 and up_vol > down_vol * 1.2 and atr5 < atr20:
                                    final_candidates.append({'Hisse': ticker, 'Puan/Durum': 'Kurumsal Birikim Sinyali', 'Fiyat': round(last_close, 2), 'RVOL': round(rvol, 2)})

                        except Exception as e:
                            continue
                            
                    if final_candidates:
                        st.success(f"Zorlu Testleri Geçen Kurumsal Adaylar ({len(final_candidates)} Adet)")
                        st.table(pd.DataFrame(final_candidates))
                    else:
                        st.warning("Piyasada bugün bu zorlu (RVOL, EMA, Kapanış Yeri) matematiksel kuralların tümünü aynı anda sağlayan bir setup (kurulum) oluşmadı. Profesyoneller gibi sabredin, zorlama işlem yapmayın.")
            
            except Exception as e:
                st.error(f"Tarama Hatası: {e}")

st.write("---")

# ==========================================
# ORTAK MODÜL: ANALİZ VE ALPACA EMİR SİSTEMİ
# ==========================================
st.subheader("Hedef/Stop Emir Merkezi")

ticker_input = st.text_input("İşlem Yapılacak Hisse Sembolü (Örn: CAR)", "").upper()

if ticker_input:
    if st.button(f"🔍 {ticker_input} İçin VWAP Analizi Yap"):
        try:
            stock = yf.Ticker(ticker_input)
            df_vwap = stock.history(period='1d', interval='5m', prepost=True)
            if not df_vwap.empty:
                df_vwap['Typical_Price'] = (df_vwap['High'] + df_vwap['Low'] + df_vwap['Close']) / 3
                df_vwap['VP'] = df_vwap['Typical_Price'] * df_vwap['Volume']
                vwap_price = round((df_vwap['VP'].cumsum() / df_vwap['Volume'].cumsum()).iloc[-1], 2)
                
                st.info(f"📊 **Analiz:** Güncel Fiyat: ${round(df_vwap['Close'].iloc[-1], 2)} | **VWAP (Referans): ${vwap_price}**")
            else:
                st.warning("Analiz için veri bulunamadı.")
        except: pass

if api_key and secret_key:
    try:
        api = get_api(api_key, secret_key)
        with st.form("bracket_order_form"):
            col1, col2 = st.columns(2)
            with col1:
                qty = st.number_input("Adet", min_value=1, value=100)
                limit_price = st.number_input("Alış Limit Fiyatı ($)", min_value=0.01, value=1.00, step=0.01)
            with col2:
                take_profit_price = st.number_input("Kar-Al Fiyatı ($) -> %15 Hedef", min_value=0.01, value=limit_price * 1.15, step=0.01)
                stop_loss_price = st.number_input("Zarar-Kes Fiyatı ($) -> %5 Risk", min_value=0.01, value=limit_price * 0.95, step=0.01)

            if st.form_submit_button("🚀 Emri Piyasaya Gönder") and ticker_input:
                api.submit_order(
                    symbol=ticker_input, qty=qty, side='buy', type='limit', time_in_force='day',
                    limit_price=limit_price, extended_hours=True, order_class='bracket',
                    take_profit=dict(limit_price=round(take_profit_price, 2)),
                    stop_loss=dict(stop_price=round(stop_loss_price, 2), limit_price=round(stop_loss_price - 0.02, 2))
                )
                st.success(f"İşlem Başarılı! {ticker_input} için {limit_price}$ seviyesinden Bracket emir iletildi.")
    except: pass
