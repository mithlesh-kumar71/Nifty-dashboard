import yfinance as yf
import pandas as pd
import streamlit as st
import ta
import plotly.graph_objects as go

# ---------------- SUPER TREND FUNCTION ----------------
def supertrend(df, period=14, multiplier=3):
    atr_indicator = ta.volatility.AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Close'], window=period
    )
    atr = atr_indicator.average_true_range()   # Series, not DataFrame

    hl2 = (df['High'] + df['Low']) / 2
    df['UpperBand'] = hl2 + (multiplier * atr)
    df['LowerBand'] = hl2 - (multiplier * atr)

    df['Supertrend'] = True
    for i in range(1, len(df)):
        if df['Close'][i] > df['UpperBand'][i-1]:
            df.loc[df.index[i], 'Supertrend'] = True
        elif df['Close'][i] < df['LowerBand'][i-1]:
            df.loc[df.index[i], 'Supertrend'] = False
        else:
            df.loc[df.index[i], 'Supertrend'] = df['Supertrend'][i-1]
            if df['Supertrend'][i] and df['LowerBand'][i] < df['LowerBand'][i-1]:
                df.loc[df.index[i], 'LowerBand'] = df['LowerBand'][i-1]
            if not df['Supertrend'][i] and df['UpperBand'][i] > df['UpperBand'][i-1]:
                df.loc[df.index[i], 'UpperBand'] = df['UpperBand'][i-1]

    return df

# ---------------- STOCKS TO TRACK ----------------
symbols_map = {
    "NIFTY": "^NSEI",
    "RELIANCE": "RELIANCE.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "SBI": "SBIN.NS",
    "TCS": "TCS.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "BEL": "BEL.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "HAL": "HAL.NS",
    "INFOSYS": "INFY.NS",
    "VEDL": "VEDL.NS"
}

# ---------------- STREAMLIT APP ----------------
st.set_page_config(layout="wide")
st.title("📊 Market Trend & Signals Dashboard")

selected_symbol = st.selectbox("Select Stock", list(symbols_map.keys()))

# Load OHLCV data
df = yf.download(symbols_map[selected_symbol], period="6mo", interval="1d")
df = supertrend(df)

# Latest Signal
latest_signal = "BUY ✅" if df['Supertrend'].iloc[-1] else "SELL ❌"
st.subheader(f"📢 Latest Signal for {selected_symbol}: {latest_signal}")

# Show last 10 rows
st.subheader("🔎 Recent Trend Data")
st.dataframe(df[['Close', 'Supertrend', 'UpperBand', 'LowerBand']].tail(10))

# ---------------- CHART ----------------
fig = go.Figure()

# Candlestick
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'],
    name="Candles"
))

# Supertrend bands
fig.add_trace(go.Scatter(
    x=df.index, y=df['UpperBand'],
    line=dict(color="red", width=1), name="UpperBand"
))
fig.add_trace(go.Scatter(
    x=df.index, y=df['LowerBand'],
    line=dict(color="green", width=1), name="LowerBand"
))

# Buy/Sell markers
buy_signals = df[(df['Supertrend'] == True) & (df['Supertrend'].shift(1) == False)]
sell_signals = df[(df['Supertrend'] == False) & (df['Supertrend'].shift(1) == True)]

fig.add_trace(go.Scatter(
    x=buy_signals.index, y=buy_signals['Close'],
    mode="markers", marker=dict(color="green", size=10, symbol="triangle-up"),
    name="BUY"
))
fig.add_trace(go.Scatter(
    x=sell_signals.index, y=sell_signals['Close'],
    mode="markers", marker=dict(color="red", size=10, symbol="triangle-down"),
    name="SELL"
))

fig.update_layout(title=f"{selected_symbol} - Supertrend Strategy",
                  xaxis_title="Date", yaxis_title="Price",
                  template="plotly_dark", height=700)

st.plotly_chart(fig, use_container_width=True)
