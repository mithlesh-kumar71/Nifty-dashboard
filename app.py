import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Supertrend Calculation ----------------
def atr(df, period=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = abs(df['High'] - df['Close'].shift())
    df['L-C'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df

def supertrend(df, period=7, multiplier=3):
    df = atr(df, period)
    
    # Ensure hl2 and ATR are Series, not DataFrames
    hl2 = ((df['High'].astype(float) + df['Low'].astype(float)) / 2).astype(float)
    atr_values = df['ATR'].astype(float)

    df['UpperBand'] = hl2 + (multiplier * atr_values)
    df['LowerBand'] = hl2 - (multiplier * atr_values)
    df['Supertrend'] = np.nan

    for i in range(period, len(df)):
        if df['Close'].iloc[i] > df['UpperBand'].iloc[i - 1]:
            df.loc[df.index[i], 'Supertrend'] = 1   # Buy signal
        elif df['Close'].iloc[i] < df['LowerBand'].iloc[i - 1]:
            df.loc[df.index[i], 'Supertrend'] = -1  # Sell signal
        else:
            df.loc[df.index[i], 'Supertrend'] = df['Supertrend'].iloc[i - 1]

    return df

# ---------------- Streamlit App ----------------
st.set_page_config(page_title="Intraday Trading Dashboard", layout="wide")

st.title("📊 Intraday Trading Signals (NIFTY & NIFTY50 Stocks)")

# Dropdown with NIFTY50 stocks
stocks = {
    "NIFTY 50": "^NSEI",
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Infosys": "INFY.NS",
    "State Bank of India": "SBIN.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Axis Bank": "AXISBANK.NS",
    "Larsen & Toubro": "LT.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Wipro": "WIPRO.NS",
    "ONGC": "ONGC.NS",
    "Power Grid": "POWERGRID.NS",
    "HCL Tech": "HCLTECH.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "NTPC": "NTPC.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Tata Steel": "TATASTEEL.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Grasim": "GRASIM.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Tech Mahindra": "TECHM.NS",
    "Cipla": "CIPLA.NS",
    "Divi's Labs": "DIVISLAB.NS",
    "Nestle India": "NESTLEIND.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "Britannia": "BRITANNIA.NS",
    "Coal India": "COALINDIA.NS",
    "SBI Life": "SBILIFE.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Dr Reddy's": "DRREDDY.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Hindalco": "HINDALCO.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "JSW Energy": "JSWENERGY.NS"
}

symbol = st.selectbox("Select Stock/Index", list(stocks.keys()))
ticker = stocks[symbol]

# Interval choice
interval = st.radio("Select Interval", ["15m", "30m", "1h", "1d"], index=0)

# Download data
st.write(f"Fetching data for **{symbol}** ({ticker}) ...")
df = yf.download(ticker, period="60d" if interval != "1d" else "1y", interval=interval)

if not df.empty:
    df = supertrend(df)

    st.subheader("📈 Price & Supertrend")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Close'], label="Close Price", color="blue")
    ax.plot(df.index, df['UpperBand'], label="Upper Band", color="red", linestyle="--")
    ax.plot(df.index, df['LowerBand'], label="Lower Band", color="green", linestyle="--")

    buy_signals = df[df['Supertrend'] == 1]
    sell_signals = df[df['Supertrend'] == -1]

    ax.scatter(buy_signals.index, buy_signals['Close'], marker="^", color="green", label="Buy Signal", alpha=1)
    ax.scatter(sell_signals.index, sell_signals['Close'], marker="v", color="red", label="Sell Signal", alpha=1)

    ax.set_title(f"{symbol} Supertrend ({interval})")
    ax.legend()
    st.pyplot(fig)

    st.subheader("🔎 Latest Signals")
    latest_signal = df['Supertrend'].iloc[-1]
    if latest_signal == 1:
        st.success("✅ Current Signal: **BUY**")
    elif latest_signal == -1:
        st.error("❌ Current Signal: **SELL**")
    else:
        st.info("⚖️ Current Signal: **Neutral**")

    st.dataframe(df.tail(20))
else:
    st.error("No data found. Try another symbol/interval.")
