import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import ta

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Market Trend & Signals", layout="wide")
st.title("📊 Market Trend & Signal App (NIFTY + Selected Stocks)")

# -------------------------
# Symbol Map (Yahoo Finance Tickers)
# -------------------------
symbols_map = {
    "NIFTY 50": "^NSEI",
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "SBI": "SBIN.NS",
    "Tata Motors": "TATAMOTORS.NS"
}

# -------------------------
# Sidebar Controls
# -------------------------
symbol_name = st.sidebar.selectbox("Select Symbol", list(symbols_map.keys()), index=0)
symbol = symbols_map[symbol_name]

interval = st.sidebar.selectbox("Interval", ["15m", "30m", "60m", "1d"], index=0)
period = st.sidebar.selectbox("Period", ["5d", "1mo", "3mo", "6mo", "1y"], index=0)
chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)

# -------------------------
# Data Loader
# -------------------------
@st.cache_data(ttl=300)
def load_ohlcv(sym, period="1mo", interval="15m"):
    df = yf.download(sym, period=period, interval=interval).reset_index()
    df.dropna(inplace=True)
    return df

df = load_ohlcv(symbol, period=period, interval=interval)
if df.empty:
    st.error("No data fetched. Try changing interval/period.")
    st.stop()

# -------------------------
# Technical Indicators
# -------------------------
def atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def supertrend(df, period=10, multiplier=3):
    out = df.copy()
    _atr = atr(out, period)
    hl2 = (out["High"] + out["Low"]) / 2
    upperband = hl2 + multiplier * _atr
    lowerband = hl2 - multiplier * _atr
    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()

    # Fix: ensure scalars using .iat[]
    for i in range(1, len(out)):
        if out["Close"].iat[i - 1] <= final_upperband.iat[i - 1]:
            final_upperband.iat[i] = min(upperband.iat[i], final_upperband.iat[i - 1])
        else:
            final_upperband.iat[i] = upperband.iat[i]

        if out["Close"].iat[i - 1] >= final_lowerband.iat[i - 1]:
            final_lowerband.iat[i] = max(lowerband.iat[i], final_lowerband.iat[i - 1])
        else:
            final_lowerband.iat[i] = lowerband.iat[i]

    trend = np.ones(len(out))
    st_line = np.zeros(len(out))
    for i in range(len(out)):
        if i == 0:
            trend[i] = 1
            st_line[i] = final_lowerband.iat[i]
        else:
            if out["Close"].iat[i] > final_upperband.iat[i - 1]:
                trend[i] = 1
            elif out["Close"].iat[i] < final_lowerband.iat[i - 1]:
                trend[i] = -1
            else:
                trend[i] = trend[i - 1]
            st_line[i] = final_lowerband.iat[i] if trend[i] == 1 else final_upperband.iat[i]

    out["Supertrend"] = st_line
    out["Trend"] = trend
    return out

df = supertrend(df)
df["RSI"] = ta.momentum.RSIIndicator(df["Close"], 14).rsi()
df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

# -------------------------
# Signal Generation
# -------------------------
df["Signal"] = "Neutral"
df.loc[(df["Trend"] == 1) & (df["RSI"] > 55) & (df["Close"] > df["VWAP"]), "Signal"] = "Buy"
df.loc[(df["Trend"] == -1) & (df["RSI"] < 45) & (df["Close"] < df["VWAP"]), "Signal"] = "Sell"

latest_signal = df["Signal"].iloc[-1]
latest_close = df["Close"].iloc[-1]

# -------------------------
# Plotting
# -------------------------
fig = go.Figure()
time_col = "Datetime" if "Datetime" in df.columns else "Date"

if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=df[time_col],
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Candlestick"
    ))
else:
    fig.add_trace(go.Scatter(
        x=df[time_col], y=df["Close"], mode="lines", name="Close"
    ))

fig.add_trace(go.Scatter(
    x=df[time_col], y=df["Supertrend"], mode="lines", name="Supertrend", line=dict(color="orange")
))
fig.add_trace(go.Scatter(
    x=df[time_col], y=df["VWAP"], mode="lines", name="VWAP", line=dict(color="cyan", dash="dot")
))
fig.update_layout(title=f"{symbol_name} - {interval} Chart", template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Signal & Latest Info
# -------------------------
st.subheader("🔑 Trading Signal")
st.metric(label=f"Latest Signal for {symbol_name}", value=latest_signal, delta=f"Close: {latest_close:.2f}")

# RSI Plot
st.subheader("📉 RSI Indicator")
fig_rsi = go.Figure([go.Scatter(x=df[time_col], y=df["RSI"], mode="lines")])
fig_rsi.add_hrect(y0=30, y1=70, fillcolor="green", opacity=0.2, line_width=0)
fig_rsi.update_layout(title="RSI (14)", template="plotly_dark")
st.plotly_chart(fig_rsi, use_container_width=True)

# -------------------------
# Show Data
# -------------------------
st.subheader("📋 Latest Data")
st.dataframe(df.tail(20))
