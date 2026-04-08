import streamlit as st
import alpaca_trade_api as tradeapi
import pandas as pd
import requests
import yfinance as yf

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Algo-Trading Terminal", layout="wide")
st.title("🎯 Hibrit Momentum & İşlem Terminali (v4.0)")

# --- YAN MENÜ: API GİRİŞİ ---
st.sidebar.header("Alpaca API (Paper)")
api_key = st.sidebar.text_input("API Key ID", type="password")
secret_key = st.sidebar.text_input("Secret Key", type="password")


def get_api(api_key, secret_key):
    return tradeapi.REST(
        key_id=api_key,
        secret_key=secret_key,
        base_url='https://paper-api.alpaca.markets',
        api_version='v2'
    )


# --- MODÜL 1: CANLI MARKET MOVERS (YAHOO JSON API) ---
st.subheader("1. Aşama: Canlı Piyasa Tarayıcı (Top Gainers)")


@st.cache_data(ttl=60)
def get_top_gainers():
    try:
        url = 'https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=false&lang=en-US&region=US&scrIds=day_gainers&count=15'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()

        quotes = data['finance']['result'][0]['quotes']
        results = []
        for q in quotes:
            results.append({
                'Hisse': q.get('symbol', ''),
                'Şirket': q.get('shortName', ''),
                'Son Fiyat ($)': round(q.get('regularMarketPrice', 0), 2),
                'Artış (%)': round(q.get('regularMarketChangePercent', 0), 2),
                'Hacim': f"{int(q.get('regularMarketVolume', 0)):,}"
            })
        return pd.DataFrame(results)
    except Exception as e:
        return pd.DataFrame()


if st.button("Piyasayı Tara / Güncelle"):
    with st.spinner("Sunucudan veriler çekiliyor..."):
        df_gainers = get_top_gainers()
        if not df_gainers.empty:
            st.dataframe(df_gainers, use_container_width=True)
        else:
            st.warning("Piyasa kapalı olduğu için liste boş olabilir.")

st.divider()

# --- MODÜL 2: TEKNİK ANALİZ (VWAP) ---
st.subheader("2. Aşama: Akıllı Analiz ve İşlem")

ticker = st.text_input("İşlem Yapılacak Hisse Sembolü (Örn: CAR)", "").upper()

if ticker:
    if st.button(f"🔍 {ticker} İçin VWAP Analizi Yap"):
        with st.spinner("Grafikler inceleniyor ve seviyeler hesaplanıyor..."):
            try:
                # 5 dakikalık veriyi pre-market dahil çekiyoruz
                stock = yf.Ticker(ticker)
                df = stock.history(period='1d', interval='5m', prepost=True)

                if not df.empty:
                    # VWAP Hesaplaması
                    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
                    df['VP'] = df['Typical_Price'] * df['Volume']
                    df['VWAP'] = df['VP'].cumsum() / df['Volume'].cumsum()

                    current_price = round(df['Close'].iloc[-1], 2)
                    vwap_price = round(df['VWAP'].iloc[-1], 2)
                    day_high = round(df['High'].max(), 2)

                    # Güvenli giriş noktası: VWAP'ın tam üzeri
                    suggested_entry = round(vwap_price * 1.01, 2)

                    st.info("### 📊 Analiz Raporu")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Güncel Fiyat", f"${current_price}")
                    col_b.metric("Gün İçi Zirve", f"${day_high}")
                    col_c.metric("VWAP (Referans)", f"${vwap_price}")

                    st.success(
                        f"**Önerilen Strateji:** Bu hisse için rastgele piyasa emri girmeyin. Fiyatın **${vwap_price}** seviyesindeki VWAP desteğine çekilmesini bekleyin. Limit alış emrinizi bu seviyeye yakın bir yere kurun.")
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
                # Kullanıcı analize göre kendi limitini yazar
                limit_price = st.number_input("Alış Limit Fiyatı ($) - VWAP'a yakın tutun", min_value=0.01, value=1.00,
                                              step=0.01)

            with col2:
                # Kâr ve zarar seviyeleri matematiksel olarak senin kurallarına göre hesaplanır
                take_profit_price = st.number_input("Kar-Al Fiyatı ($) -> %15 Hedef", min_value=0.01,
                                                    value=limit_price * 1.15, step=0.01)
                stop_loss_price = st.number_input("Zarar-Kes Fiyatı ($) -> %5 Risk", min_value=0.01,
                                                  value=limit_price * 0.95, step=0.01)

            submit_button = st.form_submit_button("🚀 Emri Piyasaya Gönder")

            if submit_button and ticker:
                try:
                    api.submit_order(
                        symbol=ticker, qty=qty, side='buy', type='limit', time_in_force='day',
                        limit_price=limit_price, extended_hours=True, order_class='bracket',
                        take_profit=dict(limit_price=round(take_profit_price, 2)),
                        stop_loss=dict(stop_price=round(stop_loss_price, 2),
                                       limit_price=round(stop_loss_price - 0.02, 2))
                    )
                    st.success(f"İşlem Başarılı! {ticker} için {limit_price}$ seviyesinden emir iletildi.")
                except Exception as e:
                    st.error(f"Emir Hatası: {e}")
    except Exception:
        pass
