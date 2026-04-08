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
st.title("🎯 Hibrit Momentum & İşlem Terminali (v6.3)")

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
# SEKME 2: KURUMSAL SWING RADAR (GENİŞLETİLMİŞ EVREN)
# ==========================================
with tab2:
    st.write("Profesyonel filtrelere göre 'Ertesi Gün' (Swing) potansiyeli yüksek adaylar:")
    
    algo_choice = st.selectbox(
        "Kullanılacak Algoritmayı Seçin:",
        [
            "A) Keskin Nişancı Breakout (RVOL≥1.5, Kapanış>%75, VWAP Üstü, Pozitif RS, Kırılıma <%2)",
            "B) İkinci Gün Koşusu (Gap-Up, RVOL>3, VWAP Üstü Kapanış)",
            "C) Kurumsal Birikim (200MA Üstü, Hacimli Alışlar, Sıkışma)"
        ]
    )
    
    if st.button("🚀 Kurumsal Taramayı Başlat (Zaman Alabilir)"):
        with st.spinner("1. Aşama: Piyasadan hacimli 250 hisse toplanıyor (Evren Genişletildi)..."):
            try:
                url = "https://scanner.tradingview.com/america/scan"
                payload = {
                    "filter": [
                        {"left": "close", "operation": "greater", "right": 2.00},
                        {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
                        {"left": "volume", "operation": "greater", "right": 300000} # Hacim barajını biraz indirdik ki daha çok hisse taransın
                    ],
                    "options": {"lang": "en"}, "markets": ["america"],
                    "symbols": {"query": {"types": ["stock"]}, "tickers": []},
                    "columns": ["name"], "sort": {"sortBy": "change", "sortOrder": "desc"},
                    "range": [0, 250] # Havuzu 80'den 250'ye çıkardık
                }
                res = requests.post(url, json=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                tickers = [item['d'][0] for item in res.json().get('data', [])]
                
                if not tickers:
                    st.warning("Hisse çekilemedi.")
                else:
                    st.write(f"2. Aşama: {len(tickers)} hisse üzerinde o zorlu 'Keskin Nişancı' algoritmik testleri uygulanıyor...")
                    
                    tickers_to_download = tickers + ["SPY"]
                    yf_data = yf.download(tickers_to_download, period="60d", progress=False)
                    
                    try:
                        spy_close = yf_data['Close']['SPY'].dropna()
                        spy_ret_10d = (spy_close.iloc[-1] - spy_close.iloc[-10]) / spy_close.iloc[-10]
                    except:
                        spy_ret_10d = 0
                    
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
                            
                            last_close = df['Close'].iloc[-1]
                            last_high = df['High'].iloc[-1]
                            last_low = df['Low'].iloc[-1]
                            last_vol = df['Volume'].iloc[-1]
                            
                            daily_vwap = (last_high + last_low + last_close) / 3
                            
                            avg_vol_10d = df['Volume'].rolling(10).mean().iloc[-2] 
                            rvol = last_vol / avg_vol_10d if avg_vol_10d > 0 else 0
                            
                            daily_range = last_high - last_low
                            closing_strength = (last_close - last_low) / daily_range if daily_range > 0 else 0
                            
                            stock_ret_10d = (last_close - df['Close'].iloc[-10]) / df['Close'].iloc[-10]
                            rs_positive = stock_ret_10d > spy_ret_10d
                            
                            high_20d = df['High'].rolling(20).max().iloc[-1]
                            breakout_dist = (high_20d - last_close) / last_close if high_20d > last_close else 0
                            
                            if "A)" in algo_choice:
                                cond_rvol = rvol >= 1.5
                                cond_close_str = closing_strength > 0.75
                                cond_vwap = last_close > daily_vwap
                                cond_rs = rs_positive
                                cond_breakout = breakout_dist <= 0.02
                                
                                if cond_rvol and cond_close_str and cond_vwap and cond_rs and cond_breakout:
                                    final_candidates.append({
                                        'Hisse': ticker, 
                                        'Durum': 'Keskin Nişancı Onayı ✅', 
                                        'Fiyat': round(last_close, 2),
                                        'RVOL': round(rvol, 2)
                                    })
                            
                            elif "B)" in algo_choice:
                                gap = (df['Open'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
                                if gap > 2.0 and rvol > 3.0 and closing_strength >= 0.70:
                                    final_candidates.append({'Hisse': ticker, 'Durum': 'İkinci Gün Potansiyeli', 'Fiyat': round(last_close, 2), 'RVOL': round(rvol, 2)})
                                    
                            elif "C)" in algo_choice:
                                sma200 = df['Close'].rolling(window=50).mean().iloc[-1] 
                                up_vol = df[df['Close'] > df['Open']]['Volume'].tail(15).mean()
                                down_vol = df[df['Close'] < df['Open']]['Volume'].tail(15).mean()
                                
                                df['TR'] = df[['High', 'Low', 'Close']].max(axis=1) - df[['High', 'Low', 'Close']].min(axis=1)
                                atr5 = df['TR'].rolling(5).mean().iloc[-1]
                                atr20 = df['TR'].rolling(20).mean().iloc[-1]
                                
                                if last_close > sma200 and up_vol > down_vol * 1.2 and atr5 < atr20:
                                    final_candidates.append({'Hisse': ticker, 'Durum': 'Kurumsal Birikim', 'Fiyat': round(last_close, 2), 'RVOL': round(rvol, 2)})

                        except Exception as e:
                            continue
                            
                    if final_candidates:
                        st.success(f"Tüm Otorite Testlerini Geçen Adaylar ({len(final_candidates)} Adet)")
                        st.table(pd.DataFrame(final_candidates))
                    else:
                        st.warning("Piyasada bugün 250 hisse taranmasına rağmen, senin kurallarına (RVOL>1.5, Kapanış Gücü>%75, RS+, VWAP Üstü, Kırılıma <%2) uyan tek bir hisse bile bulunamadı. Nakitte beklemek de kârlı bir pozisyondur.")
            
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
