import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ----------------- ATR Calculation -----------------
def atr(df, period=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-PC'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period, min_periods=1).mean()
    return df

# ----------------- Supertrend -----------------
def supertrend(df, period=7, multiplier=3):
    df = atr(df, period)
    hl2 = (df['High'] + df['Low']) / 2
    atr_values = df['ATR']

    df['UpperBand'] = hl2 + (multiplier * atr_values)
    df['LowerBand'] = hl2 - (multiplier * atr_values)
    df['Supertrend'] = np.nan

    for i in range(period, len(df)):
        if df['Close'].iloc[i] > df['UpperBand'].iloc[i - 1]:
            df.loc[df.index[i], 'Supertrend'] = 1   # Buy
        elif df['Close'].iloc[i] < df['LowerBand'].iloc[i - 1]:
            df.loc[df.index[i], 'Supertrend'] = -1  # Sell
        else:
            df.loc[df.index[i], 'Supertrend'] = df['Supertrend'].iloc[i - 1]

    return df

# ----------------- Streamlit App -----------------
st.set_page_config(page_title="NIFTY50 Options Dashboard", layout="wide")
st.title("📈 Intraday Options Trading Dashboard")

# ✅ NIFTY50 stocks
nifty50_stocks = {
    "NIFTY 50": "^NSEI",
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Infosys": "INFY.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "SBI": "SBIN.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Axis Bank": "AXISBANK.NS",
    "HCL Tech": "HCLTECH.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "Maruti": "MARUTI.NS",
    "HUL": "HINDUNILVR.NS",
    "Wipro": "WIPRO.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    # 👉 You can extend to all NIFTY50 stocks
}

# Dropdown
stock = st.selectbox("Select Stock", list(nifty50_stocks.keys()))
symbol = nifty50_stocks[stock]

# Data interval
interval = st.selectbox("Interval", ["5m", "15m", "1h", "1d"])

# Fetch OHLCV
df = yf.download(symbol, period="1mo", interval=interval)
df.dropna(inplace=True)

# Compute supertrend
df = supertrend(df)

# Last Signal
last_signal = df['Supertrend'].iloc[-1]
signal_text = "📢 BUY Signal" if last_signal == 1 else "📉 SELL Signal"

st.subheader(f"Latest Trend for {stock}: {signal_text}")

# Plot chart
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['Close'], label="Close Price", color="blue")
buy_signals = df[df['Supertrend'] == 1]
sell_signals = df[df['Supertrend'] == -1]
ax.scatter(buy_signals.index, buy_signals['Close'], marker="^", color="green", label="Buy Signal", alpha=1)
ax.scatter(sell_signals.index, sell_signals['Close'], marker="v", color="red", label="Sell Signal", alpha=1)

ax.set_title(f"{stock} Price with Supertrend Signals")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Show table
st.subheader("Recent Data")
st.dataframe(df.tail(20))
