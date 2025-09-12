import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from nsepython import nse_optionchain_scrapper

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Intraday Strategy + Options Dashboard", layout="wide")
st.title("📊 NSE Intraday Strategy + Options Dashboard")

# -------------------------
# Sidebar controls
# -------------------------
symbol = st.sidebar.text_input("Symbol (Yahoo Finance)", "^NSEI")
interval = st.sidebar.selectbox("Chart Interval", ["1m","5m","15m","60m","1d"], index=2)
chart_type = st.sidebar.radio("Chart Type", ["Candlestick","Line"], horizontal=True)
period = st.sidebar.selectbox("Historical Period", ["1d","5d","1mo","3mo","6mo","1y"], index=1)

# -------------------------
# Fetch price data
# -------------------------
@st.cache_data(ttl=60)
def fetch_price(sym, per, intv):
    df = yf.Ticker(sym).history(period=per, interval=intv).reset_index().dropna()
    df.rename(columns={"Datetime":"Date"}, inplace=True)
    return df

df = fetch_price(symbol, period, interval)
if df.empty:
    st.error("No OHLC data found. Try changing symbol or interval.")
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
    hl2 = (out["High"] + out["Low"])/2
    upperband = hl2 + multiplier*_atr
    lowerband = hl2 - multiplier*_atr
    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()
    for i in range(1, len(out)):
        final_upperband.iloc[i] = min(upperband.iloc[i], final_upperband.iloc[i-1]) if out["Close"].iloc[i-1] <= final_upperband.iloc[i-1] else upperband.iloc[i]
        final_lowerband.iloc[i] = max(lowerband.iloc[i], final_lowerband.iloc[i-1]) if out["Close"].iloc[i-1] >= final_lowerband.iloc[i-1] else lowerband.iloc[i]
    trend = np.ones(len(out))
    st_line = np.zeros(len(out))
    for i in range(len(out)):
        if i==0:
            trend[i]=1
            st_line[i]=final_lowerband.iloc[i]
        else:
            if out["Close"].iloc[i] > final_upperband.iloc[i-1]:
                trend[i]=1
            elif out["Close"].iloc[i] < final_lowerband.iloc[i-1]:
                trend[i]=-1
            else:
                trend[i]=trend[i-1]
            st_line[i]=final_lowerband.iloc[i] if trend[i]==1 else final_upperband.iloc[i]
    out["ST"]=st_line
    out["Trend"]=trend
    return out

df = supertrend(df)
df["RSI"] = pd.Series((df["Close"].diff().fillna(0)).rolling(14).mean())
df["VWAP"] = (df["Close"]*df["Volume"]).cumsum()/df["Volume"].cumsum()

# -------------------------
# Strategy signals
# -------------------------
df["Signal"]=0
df.loc[(df["Trend"]==1) & (df["RSI"]>60) & (df["Close"]>df["VWAP"]), "Signal"]=1
df.loc[(df["Trend"]==-1) & (df["RSI"]<40) & (df["Close"]<df["VWAP"]), "Signal"]=-1

# -------------------------
# Plot Price + Indicators
# -------------------------
fig = go.Figure()
if chart_type=="Candlestick":
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Candlestick"))
else:
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["ST"], mode="lines", name="Supertrend", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=df["Date"], y=df["VWAP"], mode="lines", name="VWAP", line=dict(color="cyan", dash="dot")))
fig.update_layout(title=f"{symbol} Intraday Chart", template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Fetch Option Chain
# -------------------------
st.subheader("ATM ±300 Option Chain & PCR")
if symbol.upper()=="^NSEI":
    oc = nse_optionchain_scrapper("NIFTY")
elif symbol.upper()=="^NSEBANK":
    oc = nse_optionchain_scrapper("BANKNIFTY")
else:
    oc=None

if oc:
    underlying = oc['records']['underlyingValue']
    data = oc['records']['data']
    strikes, ce_oi, pe_oi, ce_chng, pe_chng = [],[],[],[],[]
    for d in data:
        strikes.append(d['strikePrice'])
        ce = d.get('CE', {})
        pe = d.get('PE', {})
        ce_oi.append(ce.get('openInterest',0))
        pe_oi.append(pe.get('openInterest',0))
        ce_chng.append(ce.get('changeinOpenInterest',0))
        pe_chng.append(pe.get('changeinOpenInterest',0))
    df_oc = pd.DataFrame({
        "Strike Price": strikes,
        "CE_OI": ce_oi,
        "PE_OI": pe_oi,
        "CE_Chng_OI": ce_chng,
        "PE_Chng_OI": pe_chng
    })
    df_oc["PE/CE_Chng_OI_Ratio"] = (df_oc["PE_Chng_OI"]/df_oc["CE_Chng_OI"]).replace(np.inf, np.nan).round(2)
    
    atm_strike = df_oc.iloc[(df_oc["Strike Price"] - underlying).abs().argsort()[:1]]["Strike Price"].values[0]
    df_oc_filtered = df_oc[(df_oc["Strike Price"] >= atm_strike-300) & (df_oc["Strike Price"] <= atm_strike+300)]
    
    # Highlight ATM strike
    def highlight_atm(row):
        return ['background-color: yellow' if row["Strike Price"]==atm_strike else '' for _ in row]
    st.dataframe(df_oc_filtered.style.apply(highlight_atm, axis=1).format("{:.2f}"))
    
    # PCR and PE/CE OI ratio
    pcr = df_oc_filtered["PE_OI"].sum()/max(df_oc_filtered["CE_OI"].sum(),1)
    latest_ratio = df_oc_filtered.loc[df_oc_filtered["Strike Price"]==atm_strike,"PE/CE_Chng_OI_Ratio"].values[0]
    st.metric("PCR", f"{pcr:.2f}")
    st.metric("ATM PE/CE Change OI Ratio", f"{latest_ratio:.2f}")
    
    # -------------------------
    # Line chart for ATM PE/CE change OI ratio
    # -------------------------
    st.subheader("ATM PE/CE Change OI Ratio Over Strikes")
    fig_ratio = go.Figure()
    fig_ratio.add_trace(go.Scatter(
        x=df_oc_filtered["Strike Price"],
        y=df_oc_filtered["PE/CE_Chng_OI_Ratio"],
        mode="lines+markers",
        line=dict(color="orange"),
        name="PE/CE Chng OI Ratio"
    ))
    fig_ratio.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="Neutral (1)", annotation_position="bottom right")
    fig_ratio.update_layout(template="plotly_dark", yaxis_title="PE/CE Change OI Ratio")
    st.plotly_chart(fig_ratio, use_container_width=True)

# -------------------------
# Historical data
# -------------------------
st.subheader("Historical Data & Download")
st.dataframe(df.tail(50))
st.download_button("Download CSV", df.to_csv(index=False).encode(), f"{symbol}_data.csv","text/csv")
