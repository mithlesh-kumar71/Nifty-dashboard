import streamlit as st
import yfinance as yf
import pandas as pd
import ta

st.set_page_config(page_title="Simplified Nifty & Stocks Dashboard", layout="wide")
st.title("Simplified Nifty & Stock Dashboard")

# Define the stock symbols
symbols = {
    "Nifty50": "^NSEI",
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "SBI": "SBIN.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "Tata Motors": "TATAMOTORS.NS"
}

# Select stock
stock_name = st.selectbox("Select Stock", list(symbols.keys()))
ticker = symbols[stock_name]

# Fetch historical daily data
df = yf.download(ticker, period="6mo", interval="1d")
df.reset_index(inplace=True)

st.subheader(f"{stock_name} OHLC Data")
st.dataframe(df.tail(10))  # Show last 10 rows

# Calculate Supertrend
def supertrend(df, period=7, multiplier=3):
    df = df.copy()
    atr_indicator = ta.volatility.AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Close'], window=period
    )
    df['ATR'] = atr_indicator.average_true_range()
    
    hl2 = (df['High'] + df['Low']) / 2
    df['UpperBand'] = hl2 + (multiplier * df['ATR'])
    df['LowerBand'] = hl2 - (multiplier * df['ATR'])
    df['Trend'] = True  # Default trend
    
    for i in range(1, len(df)):
        if df['Close'][i] > df['UpperBand'][i-1]:
            df['Trend'][i] = True
        elif df['Close'][i] < df['LowerBand'][i-1]:
            df['Trend'][i] = False
        else:
            df['Trend'][i] = df['Trend'][i-1]
            if df['Trend'][i] and df['LowerBand'][i] < df['LowerBand'][i-1]:
                df['LowerBand'][i] = df['LowerBand'][i-1]
            if not df['Trend'][i] and df['UpperBand'][i] > df['UpperBand'][i-1]:
                df['UpperBand'][i] = df['UpperBand'][i-1]
    return df

df = supertrend(df)

st.subheader(f"{stock_name} Supertrend")
st.dataframe(df[['Date','Close','UpperBand','LowerBand','Trend']].tail(10))
