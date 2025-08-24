import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import plotly.graph_objects as go
import ta  # technical analysis library

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Options Dashboard", layout="wide")

st.title("📊 Options Monitoring Dashboard")

# -------------------------
# Sidebar controls
# -------------------------
symbol = st.sidebar.text_input("Enter Symbol (e.g., NIFTY, RELIANCE.BO)", "NIFTY")
interval = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=2)
chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"])
refresh_rate = st.sidebar.slider("Auto Refresh (sec)", 30, 300, 30)
period = st.sidebar.selectbox("Data Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=1)

# -------------------------
# Auto refresh
# -------------------------
st_autorefresh = st.experimental_autorefresh(interval=refresh_rate * 1000, key="refresh")

# -------------------------
# Fetch data
# -------------------------
@st.cache_data(ttl=60)
def load_data(symbol, period, interval):
    ticker = yf.Ticker(symbol + ".NS") if not symbol.endswith(".BO") else yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    df = df.reset_index()
    return df

df = load_data(symbol, period, interval)

if df.empty:
    st.error("No data found. Try another symbol or timeframe.")
    st.stop()

# -------------------------
# Indicators
# -------------------------
def atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def supertrend(df, period=10, multiplier=3):
    out = df.copy()
    _atr = atr(out, period=period)
    hl2 = (out["High"] + out["Low"]) / 2.0

    upperband = hl2 + multiplier * _atr
    lowerband = hl2 - multiplier * _atr
    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()

    for i in range(1, len(out)):
        if (upperband.iloc[i] < final_upperband.iloc[i - 1]) or (out["Close"].iloc[i - 1] > final_upperband.iloc[i - 1]):
            final_upperband.iloc[i] = upperband.iloc[i]
        else:
            final_upperband.iloc[i] = final_upperband.iloc[i - 1]

        if (lowerband.iloc[i] > final_lowerband.iloc[i - 1]) or (out["Close"].iloc[i - 1] < final_lowerband.iloc[i - 1]):
            final_lowerband.iloc[i] = lowerband.iloc[i]
        else:
            final_lowerband.iloc[i] = final_lowerband.iloc[i - 1]

    trend = np.ones(len(out), dtype=int)
    st_line = np.zeros(len(out))

    for i in range(len(out)):
        if i == 0:
            trend[i] = 1
            st_line[i] = final_lowerband.iloc[i]
        else:
            if out["Close"].iloc[i] > final_upperband.iloc[i - 1]:
                trend[i] = 1
            elif out["Close"].iloc[i] < final_lowerband.iloc[i - 1]:
                trend[i] = -1
            else:
                trend[i] = trend[i - 1]

            if trend[i] == 1:
                st_line[i] = final_lowerband.iloc[i]
            else:
                st_line[i] = final_upperband.iloc[i]

    out["ST"] = st_line
    out["Trend"] = trend
    return out

df = supertrend(df)

# RSI & ADX using ta library
df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
df["ADX"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14).adx()

# -------------------------
# Plot
# -------------------------
fig = go.Figure()

if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(x=df["Date"],
                                 open=df["Open"],
                                 high=df["High"],
                                 low=df["Low"],
                                 close=df["Close"],
                                 name="Candlestick"))
else:
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Line Chart"))

# Supertrend
fig.add_trace(go.Scatter(x=df["Date"], y=df["ST"], mode="lines", line=dict(color="orange", width=1.5), name="Supertrend"))

# Layout
fig.update_layout(title=f"{symbol} Price Chart ({interval})",
                  xaxis_title="Date",
                  yaxis_title="Price",
                  template="plotly_dark",
                  height=600)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# RSI and ADX Charts
# -------------------------
col1, col2 = st.columns(2)

with col1:
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], mode="lines", name="RSI"))
    fig_rsi.add_hrect(y0=30, y1=70, fillcolor="green", opacity=0.2, line_width=0)
    fig_rsi.update_layout(title="RSI (14)", height=300, template="plotly_dark")
    st.plotly_chart(fig_rsi, use_container_width=True)

with col2:
    fig_adx = go.Figure()
    fig_adx.add_trace(go.Scatter(x=df["Date"], y=df["ADX"], mode="lines", name="ADX", line=dict(color="cyan")))
    fig_adx.update_layout(title="ADX (14)", height=300, template="plotly_dark")
    st.plotly_chart(fig_adx, use_container_width=True)

# -------------------------
# Historical Data
# -------------------------
st.subheader("📜 Historical Data")
st.dataframe(df.tail(50))

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download Full Data as CSV", csv, f"{symbol}_data.csv", "text/csv")
