import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import ta
from nsepython import nse_optionchain_scrapper
import math

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="Intraday Options Dashboard", layout="wide")
st.title("📊 Intraday Options Trading Dashboard")

# -------------------------
# Sidebar Controls
# -------------------------
symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"], index=0)
interval = st.sidebar.selectbox("Chart Interval", ["5m", "15m"], index=1)
period = st.sidebar.selectbox("Historical Period", ["1d", "5d", "1mo"], index=0)

# -------------------------
# Fetch Price Data
# -------------------------
@st.cache_data(ttl=60)
def fetch_price(sym, per, intv):
    sym_yf = "^NSEI" if sym == "NIFTY" else "^NSEBANK"
    df = yf.Ticker(sym_yf).history(period=per, interval=intv).reset_index().dropna()
    df.rename(columns={"Datetime": "Date"}, inplace=True)
    return df

df = fetch_price(symbol, period, interval)

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
            min(upperband.iloc[i], final_upperband.iloc[i-1])
            if out["Close"].iloc[i-1] <= final_upperband.iloc[i-1]
            else upperband.iloc[i]
        )
        final_lowerband.iloc[i] = (
            max(lowerband.iloc[i], final_lowerband.iloc[i-1])
            if out["Close"].iloc[i-1] >= final_lowerband.iloc[i-1]
            else lowerband.iloc[i]
        )

    trend = np.ones(len(out))
    st_line = np.zeros(len(out))
    for i in range(len(out)):
        if i == 0:
            trend[i] = 1
            st_line[i] = final_lowerband.iloc[i]
        else:
            if out["Close"].iloc[i] > final_upperband.iloc[i-1]:
                trend[i] = 1
            elif out["Close"].iloc[i] < final_lowerband.iloc[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
            st_line[i] = final_lowerband.iloc[i] if trend[i] == 1 else final_upperband.iloc[i]

    out["ST"] = st_line
    out["Trend"] = trend
    return out

df = supertrend(df)
df["RSI"] = ta.momentum.RSIIndicator(df["Close"], 14).rsi()
df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

# -------------------------
# Price Chart
# -------------------------
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"],
                             low=df["Low"], close=df["Close"], name="Candlestick"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["VWAP"], mode="lines", name="VWAP", line=dict(color="cyan", dash="dot")))
fig.add_trace(go.Scatter(x=df["Date"], y=df["ST"], mode="lines", name="Supertrend", line=dict(color="orange")))
fig.update_layout(title=f"{symbol} Intraday Chart", template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# RSI Chart
# -------------------------
fig_rsi = go.Figure([go.Scatter(x=df["Date"], y=df["RSI"], mode="lines")])
fig_rsi.add_hrect(y0=30, y1=70, fillcolor="green", opacity=0.2, line_width=0)
fig_rsi.update_layout(title="RSI (14)", template="plotly_dark")
st.plotly_chart(fig_rsi, use_container_width=True)

# -------------------------
# Option Chain + PCR
# -------------------------
st.subheader("📑 Live Option Chain & PCR")

try:
    oc = nse_optionchain_scrapper(symbol)
    oc_df = pd.DataFrame([
        {
            "Strike Price": d["strikePrice"],
            "CE_OI": d.get("CE", {}).get("openInterest", 0),
            "PE_OI": d.get("PE", {}).get("openInterest", 0),
            "CE_Change_OI": d.get("CE", {}).get("changeinOpenInterest", 0),
            "PE_Change_OI": d.get("PE", {}).get("changeinOpenInterest", 0),
        }
        for d in oc["records"]["data"]
    ])

    atm_strike = round(oc["records"]["underlyingValue"] / 50) * 50
    df_filtered = oc_df[(oc_df["Strike Price"] >= atm_strike - 300) & (oc_df["Strike Price"] <= atm_strike + 300)]
    df_filtered["PE/CE_Chng_OI_Ratio"] = np.where(
        df_filtered["CE_Change_OI"] != 0,
        df_filtered["PE_Change_OI"] / df_filtered["CE_Change_OI"],
        np.nan
    )

    pcr = df_filtered["PE_OI"].sum() / max(df_filtered["CE_OI"].sum(), 1)

    st.write(f"**Underlying Value**: {oc['records']['underlyingValue']} | **PCR**: {pcr:.2f}")
    st.dataframe(df_filtered.style.format(precision=2))

    # -------------------------
    # PCR Dial
    # -------------------------
    latest_ratio = df_filtered.loc[df_filtered["Strike Price"] == atm_strike, "PE/CE_Chng_OI_Ratio"].values[0]

    dial_color = "yellow"
    if latest_ratio < 1:
        dial_color = "red"
    elif latest_ratio > 1:
        dial_color = "green"

    fig_dial = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_ratio,
        title={'text': "ATM PCR Change OI"},
        gauge={'axis': {'range': [0, 3]},
               'bar': {'color': dial_color},
               'steps': [
                   {'range': [0, 1], 'color': "red"},
                   {'range': [1, 1.2], 'color': "yellow"},
                   {'range': [1.2, 3], 'color': "green"}
               ]}
    ))
    st.plotly_chart(fig_dial, use_container_width=True)

except Exception as e:
    st.error(f"Error fetching Option Chain: {e}")

# -------------------------
# Download Data
# -------------------------
st.subheader("⬇️ Download Data")
csv = df.to_csv(index=False).encode()
st.download_button("Download Price Data (CSV)", csv, f"{symbol}_price.csv", "text/csv")

excel = df_filtered.to_csv(index=False).encode()
st.download_button("Download Option Chain (CSV)", excel, f"{symbol}_options.csv", "text/csv")
