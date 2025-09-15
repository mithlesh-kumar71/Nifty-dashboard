import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------- SUPER TREND STRATEGY -----------------
def supertrend(df, period=7, multiplier=3):
    hl2 = (df["High"] + df["Low"]) / 2
    df["ATR"] = df["High"].rolling(period).max() - df["Low"].rolling(period).min()
    df["UpperBand"] = hl2 + (multiplier * df["ATR"])
    df["LowerBand"] = hl2 - (multiplier * df["ATR"])
    df["Supertrend"] = np.nan

    for i in range(period, len(df)):
        if df["Close"].iloc[i] > df["UpperBand"].iloc[i - 1]:
            df.loc[df.index[i], "Supertrend"] = 1  # Buy
        elif df["Close"].iloc[i] < df["LowerBand"].iloc[i - 1]:
            df.loc[df.index[i], "Supertrend"] = -1  # Sell
        else:
            df.loc[df.index[i], "Supertrend"] = df["Supertrend"].iloc[i - 1]

    return df

# ----------------- APP -----------------
st.set_page_config(page_title="Trading Signals", layout="wide")
st.title("📊 Intraday Trading Signal App (NIFTY 50 + NIFTY Index)")

# Mapping of NIFTY 50 Stocks with Yahoo Finance tickers
symbols = {
    "NIFTY": "^NSEI",
    "Reliance": "RELIANCE.NS", "TCS": "TCS.NS", "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS", "SBI": "SBIN.NS",
    "Kotak Bank": "KOTAKBANK.NS", "Axis Bank": "AXISBANK.NS", "HDFC": "HDFC.NS",
    "Bajaj Finance": "BAJFINANCE.NS", "Bajaj Finserv": "BAJAJFINSV.NS",
    "Hindustan Unilever": "HINDUNILVR.NS", "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS", "Asian Paints": "ASIANPAINT.NS",
    "Tata Motors": "TATAMOTORS.NS", "Tata Steel": "TATASTEEL.NS",
    "JSW Steel": "JSWSTEEL.NS", "UltraTech Cement": "ULTRACEMCO.NS",
    "Maruti Suzuki": "MARUTI.NS", "Mahindra & Mahindra": "M&M.NS",
    "Bharti Airtel": "BHARTIARTL.NS", "HCL Tech": "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS", "Wipro": "WIPRO.NS",
    "Power Grid": "POWERGRID.NS", "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS", "Coal India": "COALINDIA.NS",
    "Adani Ports": "ADANIPORTS.NS", "Adani Enterprises": "ADANIENT.NS",
    "Grasim": "GRASIM.NS", "Nestle India": "NESTLEIND.NS",
    "Cipla": "CIPLA.NS", "Sun Pharma": "SUNPHARMA.NS",
    "Divi's Lab": "DIVISLAB.NS", "Dr. Reddy": "DRREDDY.NS",
    "Britannia": "BRITANNIA.NS", "Eicher Motors": "EICHERMOT.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS", "Titan": "TITAN.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS", "SBI Life": "SBILIFE.NS",
    "HDFC Life": "HDFCLIFE.NS", "ICICI Lombard": "ICICIGI.NS",
    "IndusInd Bank": "INDUSINDBK.NS", "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Shree Cement": "SHREECEM.NS", "UPL": "UPL.NS"
}

# User input
choice = st.selectbox("Choose Symbol", list(symbols.keys()))
interval = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h"])
df = yf.download(symbols[choice], period="30d", interval=interval)

if not df.empty:
    df = supertrend(df)

    # Plot chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], label="Close Price")
    buy_signals = df[df["Supertrend"] == 1]
    sell_signals = df[df["Supertrend"] == -1]
    ax.scatter(buy_signals.index, buy_signals["Close"], marker="^", color="g", label="Buy", s=100)
    ax.scatter(sell_signals.index, sell_signals["Close"], marker="v", color="r", label="Sell", s=100)
    ax.legend()
    st.pyplot(fig)

    # Show latest signal
    latest_signal = df["Supertrend"].iloc[-1]
    latest_price = df["Close"].iloc[-1]

    if latest_signal == 1:
        st.success(f"✅ BUY Signal for {choice} at {latest_price:.2f}")
    elif latest_signal == -1:
        st.error(f"❌ SELL Signal for {choice} at {latest_price:.2f}")
    else:
        st.warning(f"⚠️ No clear signal for {choice}")
else:
    st.error("No data available. Try a different symbol/interval.")
