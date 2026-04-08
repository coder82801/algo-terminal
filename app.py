import os
import streamlit as st
import alpaca_trade_api as tradeapi
import pandas as pd
import requests
import yfinance as yf

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Algo-Trading Terminal", layout="wide")
st.title("🎯 Hibrit Momentum & İşlem Terminali (v4.6)")

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

# --- MODÜL 1: CANLI MARKET MOVERS (ÇİFT MOTORLU HİBRİT RADAR) ---
st.subheader("1. Aşama: Canlı Piyasa Tarayıcı (Top Gainers)")
st.write("Hedefler TradingView'dan, CANLI fiyatlar Yahoo'dan çekiliyor:")

@st.cache_data(ttl=30) # Canlılık için yenileme süresini 30 saniyeye indirdik
def get_top_gainers():
    try:
        # ADIM 1: TradingView'dan sadece hisse sembollerini bul (Tarama)
        url_tv = "https://scanner.tradingview.com/america/scan"
        payload = {
            "filter": [
                {"left": "volume", "operation": "greater", "right": 50000}, 
                {"left": "close", "operation": "greater", "right": 0.50}, 
                {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]} 
            ],
            "options": {"lang": "en"},
            "markets": ["america"],
            "symbols": {"query": {"types": ["stock"]}, "tickers": []},
            "columns": ["name"], # Sadece isimleri istiyoruz
            "sort": {"sortBy": "change", "sortOrder": "desc"},
            "range": [0, 20] 
        }
        
        headers_tv = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Content-Type": "application/json"
        }
        
        res_tv = requests.post(url_tv, json=payload, headers=headers_tv, timeout=10)
        res_tv.raise_for_status() 
        tv_data = res_tv.json()

        # Sembolleri bir listeye al (Örn: ['UCAR', 'CAR', 'AEHR'])
        tickers = [item['d'][0] for item in tv_data.get('data', [])]
        
        if not tickers:
            return pd.DataFrame()

        # ADIM 2: Bulunan sembollerin %100 CANLI fiyatlarını Yahoo Finance'tan çek
        tickers_str = ",".join(tickers)
        url_yf = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={tickers_str}"
        headers_yf = {'User-Agent': 'Mozilla/5.0'}
        
        res_yf = requests.get(url_yf, headers=headers_yf, timeout=10)
        yf_data = res_yf.json()
        
        results = []
        for quote in yf_data['quoteResponse']['result']:
            fiyat = quote.get('regularMarketPrice', 0)
            fiyat_format = round(fiyat, 4) if fiyat < 1 else round(fiyat, 2)
            
            results.append({
                'Hisse': quote.get('symbol', ''),
                'Şirket': quote.get('shortName', ''),
                'Son Fiyat ($)': fiyat_format,
                'Artış (%)': round(quote.get('regularMarketChangePercent', 0), 2),
                'Hacim': f"{int(quote.get('regularMarketVolume', 0)):,}"
            })
            
        # Tabloyu artış yüzdesine göre yeniden büyükten küçüğe sırala
        df = pd.DataFrame(results)
        df = df.sort_values(by='Artış (%)', ascending=False).reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"Sistem Hatası (Log): {e}")
        return pd.DataFrame()

if st.button("Piyasayı Tara / Güncelle"):
    with st.spinner("Piyasa taranıyor ve canlı fiyatlar çekiliyor..."):
        df_gainers = get_top_gainers()
        if not df_gainers.empty:
            st.dataframe(df_gainers, use_container_width=True)
        else:
            st.warning("Şu an veri çekilemedi. Piyasa kapalı olabilir veya bağlantı sorunu var.")

st.divider()

# --- MODÜL 2: TEKNİK ANALİZ (VWAP) ---
st.subheader("2. Aşama: Akıllı Analiz ve İşlem")

ticker = st.text_input("İşlem Yapılacak Hisse Sembolü (Örn: UCAR)", "").upper()

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
                    col_a.metric("Güncel Fiyat", f"${current_price}")
                    col_b.metric("Gün İçi Zirve", f"${day_high}")
                    col_c.metric("VWAP (Referans)", f"${vwap_price}")
                    
                    st.success(f"**Önerilen Strateji:** Rastgele piyasa emri girmeyin. Fiyatın **${vwap_price}** seviyesindeki VWAP desteğine çekilmesini bekleyin. Limit alış emrinizi bu seviyeye yakın kurun.")
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
