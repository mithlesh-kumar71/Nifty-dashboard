import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import ta
from datetime import datetime
from nsepython import option_chain

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Intraday Strategy Dashboard", layout="wide")
st.title("📊 NSE Intraday Strategy + Options Dashboard")

# -------------------------
# Sidebar controls
# -------------------------
symbol = st.sidebar.text_input("Symbol (Yahoo Finance)", "^NSEI")
interval = st.sidebar.selectbox("Chart Interval", ["1m", "5m", "15m", "60m", "1d"], index=2)
chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)
period = st.sidebar.selectbox("Historical Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=1)

# -------------------------
# Fetch price data
# -------------------------
@st.cache_data(ttl=60)
def fetch_price(sym, per, intv):
    df = yf.Ticker(sym).history(period=per, interval=intv).reset_index().dropna()
    df.rename(columns={"Datetime": "Date"}, inplace=True)
    return df

df = fetch_price(symbol, period, interval)
if df.empty:
    st.error("No OHLC data found. Try changing symbol or interval.")
    st.stop()

# -------------------------
# Indicators
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

    for i in range(1, len(out)):
        final_upperband.iloc[i] = min(
            upperband.iloc[i], final_upperband.iloc[i - 1]
        ) if out["Close"].iloc[i - 1] <= final_upperband.iloc[i - 1] else upperband.iloc[i]
        final_lowerband.iloc[i] = max(
            lowerband.iloc[i], final_lowerband.iloc[i - 1]
        ) if out["Close"].iloc[i - 1] >= final_lowerband.iloc[i - 1] else lowerband.iloc[i]

    trend = np.ones(len(out))
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
            st_line[i] = final_lowerband.iloc[i] if trend[i] == 1 else final_upperband.iloc[i]

    out["]()
