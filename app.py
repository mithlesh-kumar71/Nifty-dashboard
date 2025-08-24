import streamlit as st
import pandas as pd, numpy as np
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import requests
import ta  # technical indicators

st.set_page_config(page_title="Options Dashboard", layout="wide")
st.title("📊 NSE Options + Charts Dashboard")

# Sidebar
symbol = st.sidebar.text_input("Symbol (Yahoo Finance)", "^NSEI")
interval = st.sidebar.selectbox("Chart Interval", ["1m", "5m", "15m", "60m", "1d"], index=2)
chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)
refresh_sec = st.sidebar.slider("Refresh every (sec)", 30, 300, 60, 30)
period = st.sidebar.selectbox("Historical Period", ["1d","5d","1mo","3mo","6mo","1y"], index=1)

st_autorefresh(interval=refresh_sec * 1000, key="refresh")

# Fetch OHLC data
@st.cache_data(ttl=60)
def fetch_price(sym, per, intv):
    try:
        df = yf.Ticker(sym).history(period=per, interval=intv).reset_index().dropna()
        if "Datetime" not in df.columns:
            df.rename(columns={"Date": "Datetime"}, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

df = fetch_price(symbol, period, interval)
if df.empty:
    st.error("No OHLC data found — try changing symbol or interval.")
    st.stop()

# ---------------- Supertrend ----------------
def atr(df, period=14):
    df["H-L"] = df["High"] - df["Low"]
    df["H-C"] = (df["High"] - df["Close"].shift()).abs()
    df["L-C"] = (df["Low"] - df["Close"].shift()).abs()
    df["TR"] = df[["H-L","H-C","L-C"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(period).mean()
    return df

def supertrend(df, period=10, multiplier=3):
    df = atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    df["upperband"] = hl2 + (multiplier * df["ATR"])
    df["lowerband"] = hl2 - (multiplier * df["ATR"])
    df["in_uptrend"] = True
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["upperband"].iloc[i-1]:
            df.at[i,"in_uptrend"] = True
        elif df["Close"].iloc[i] < df["lowerband"].iloc[i-1]:
            df.at[i,"in_uptrend"] = False
        else:
            df.at[i,"in_uptrend"] = df["in_uptrend"].iloc[i-1]
            if df["in_uptrend"].iloc[i] and df["lowerband"].iloc[i] < df["lowerband"].iloc[i-1]:
                df.at[i,"lowerband"] = df["lowerband"].iloc[i-1]
            if not df["in_uptrend"].iloc[i] and df["upperband"].iloc[i] > df["upperband"].iloc[i-1]:
                df.at[i,"upperband"] = df["upperband"].iloc[i-1]
    df["ST"] = np.where(df["in_uptrend"], df["lowerband"], df["upperband"])
    return df

df = supertrend(df)
df["RSI"] = ta.momentum.RSIIndicator(df["Close"], 14).rsi()
df["ADX"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], 14).adx()

# ---------------- Price Chart ----------------
fig = go.Figure()
if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=df["Datetime"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Candles"
    ))
else:
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["Close"], mode="lines", name="Close"))

fig.add_trace(go.Scatter(x=df["Datetime"], y=df["ST"], mode="lines", name="Supertrend", line=dict(color="orange")))
fig.update_layout(title=f"{symbol} ({interval})", template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ---------------- RSI & ADX ----------------
r1, r2 = st.columns(2)
with r1:
    fig_rsi = go.Figure([go.Scatter(x=df["Datetime"], y=df["RSI"], mode="lines", name="RSI")])
    fig_rsi.add_hrect(y0=30, y1=70, fillcolor="green", opacity=0.2, line_width=0)
    fig_rsi.update_layout(title="RSI (14)", template="plotly_dark")
    st.plotly_chart(fig_rsi, use_container_width=True)

with r2:
    fig_adx = go.Figure([go.Scatter(x=df["Datetime"], y=df["ADX"], mode="lines", name="ADX")])
    fig_adx.update_layout(title="ADX (14)", template="plotly_dark")
    st.plotly_chart(fig_adx, use_container_width=True)

# ---------------- Option Chain ----------------
st.subheader("📌 Live Option Chain & Open Interest")

def fetch_option_chain(idx):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={idx}"
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com"})
    s.get("https://www.nseindia.com", timeout=10)
    resp = s.get(url, timeout=15); resp.raise_for_status()
    return resp.json()

symbol_key = "NIFTY" if symbol.upper().startswith("N") else "BANKNIFTY"
try:
    ocj = fetch_option_chain(symbol_key)
    underlying = ocj["records"]["underlyingValue"]
    expiry = ocj["records"]["expiryDates"][0]
    data = ocj["records"]["data"]

    df_oc = pd.DataFrame([
        {"strike": d["strikePrice"],
         "CE_OI": d.get("CE", {}).get("openInterest", 0),
         "PE_OI": d.get("PE", {}).get("openInterest", 0),
         "CE_LTP": d.get("CE", {}).get("lastPrice"),
         "PE_LTP": d.get("PE", {}).get("lastPrice")}
        for d in data
    ])

    pcr = df_oc["PE_OI"].sum() / max(df_oc["CE_OI"].sum(), 1)
    atm_strike = df_oc.iloc[(df_oc["strike"] - underlying).abs().argsort()[:1]]["strike"].values[0]
    ce_ltp = df_oc.loc[df_oc["strike"] == atm_strike, "CE_LTP"].values[0]
    pe_ltp = df_oc.loc[df_oc["strike"] == atm_strike, "PE_LTP"].values[0]

    st.write(f"**Underlying:** {underlying:.2f} | **Expiry:** {expiry}")
    st.write(f"**PCR:** {pcr:.2f} | **ATM Strike:** {atm_strike} | CE LTP: {ce_ltp} | PE LTP: {pe_ltp}")
    st.dataframe(df_oc.sort_values("strike").head(20))
except Exception as e:
    st.error(f"Failed to fetch option chain: {e}")

# ---------------- Historical Data ----------------
st.subheader("📑 Historical Data & Download")
st.dataframe(df.tail(50))
st.download_button("Download CSV", df.to_csv(index=False).encode(), f"{symbol}_data.csv", "text/csv")
