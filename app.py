import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import ta
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
refresh_sec = st.sidebar.slider("Refresh every (sec)", 30, 300, 60, 30)
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
        final_upperband.iloc[i] = (
            min(upperband.iloc[i], final_upperband.iloc[i - 1])
            if out["Close"].iloc[i - 1] <= final_upperband.iloc[i - 1]
            else upperband.iloc[i]
        )
        final_lowerband.iloc[i] = (
            max(lowerband.iloc[i], final_lowerband.iloc[i - 1])
            if out["Close"].iloc[i - 1] >= final_lowerband.iloc[i - 1]
            else lowerband.iloc[i]
        )

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

    out["ST"] = st_line
    out["Trend"] = trend
    return out

df = supertrend(df)
df["RSI"] = ta.momentum.RSIIndicator(df["Close"], 14).rsi()
df["ADX"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], 14).adx()
df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

# -------------------------
# Strategy signals
# -------------------------
df["Signal"] = 0
df.loc[(df["Trend"] == 1) & (df["RSI"] > 60) & (df["Close"] > df["VWAP"]), "Signal"] = 1  # Buy
df.loc[(df["Trend"] == -1) & (df["RSI"] < 40) & (df["Close"] < df["VWAP"]), "Signal"] = -1  # Sell

# -------------------------
# Plotting Price + Indicators
# -------------------------
fig = go.Figure()
if chart_type == "Candlestick":
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candlestick",
        )
    )
else:
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))

fig.add_trace(go.Scatter(x=df["Date"], y=df["ST"], mode="lines", name="Supertrend", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=df["Date"], y=df["VWAP"], mode="lines", name="VWAP", line=dict(color="cyan", dash="dot")))
fig.update_layout(title=f"{symbol} Intraday Chart", template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# RSI & ADX
# -------------------------
c1, c2 = st.columns(2)
with c1:
    fig_rsi = go.Figure([go.Scatter(x=df["Date"], y=df["RSI"], mode="lines")])
    fig_rsi.add_hrect(y0=30, y1=70, fillcolor="green", opacity=0.2, line_width=0)
    fig_rsi.update_layout(title="RSI (14)", template="plotly_dark")
    st.plotly_chart(fig_rsi, use_container_width=True)
with c2:
    fig_adx = go.Figure([go.Scatter(x=df["Date"], y=df["ADX"], mode="lines")])
    fig_adx.update_layout(title="ADX (14)", template="plotly_dark")
    st.plotly_chart(fig_adx, use_container_width=True)

# -------------------------
# Option Chain & PCR (using nsepython)
# -------------------------
st.subheader("Live Option Chain & PCR")

# Yahoo Finance -> NSE option chain mapping
yahoo_to_nse = {
    "^NSEI": "NIFTY",
    "^NSEBANK": "BANKNIFTY",
    "^NSEFIN": "FINNIFTY",
    "^MIDCPNIFTY": "MIDCPNIFTY",
}

symbol_key = yahoo_to_nse.get(symbol.upper(), None)

if symbol_key:
    try:
        oc = option_chain(symbol_key)
        underlying = oc["records"]["underlyingValue"]
        data = oc["records"]["data"]

        df_oc = pd.DataFrame(
            [
                {
                    "strike": d["strikePrice"],
                    "CE_OI": d.get("CE", {}).get("openInterest", 0),
                    "PE_OI": d.get("PE", {}).get("openInterest", 0),
                }
                for d in data
            ]
        )

        pcr = df_oc["PE_OI"].sum() / max(df_oc["CE_OI"].sum(), 1)
        st.metric("PCR", f"{pcr:.2f}")
        st.write(f"Underlying: {underlying:.2f}")
        st.dataframe(df_oc.sort_values("strike").head(20))

    except Exception as e:
        st.warning(f"Error fetching Option Chain: {e}")
else:
    st.info("Option Chain only available for NIFTY (^NSEI), BANKNIFTY (^NSEBANK), FINNIFTY (^NSEFIN), MIDCPNIFTY (^MIDCPNIFTY).")

# -------------------------
# Historical data & download
# -------------------------
st.subheader("Historical Data & Download")
st.dataframe(df.tail(50))
csv = df.to_csv(index=False).encode()
st.download_button("Download CSV", csv, f"{symbol}_data.csv", "text/csv")
