# app.py
# Intraday Options Dashboard — Extended to Stocks (Reliance, TCS, SBI, etc.)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import math

from nsepython import nse_optionchain_scrapper

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Intraday Options + OI Heatmap", layout="wide")

# -------------------------
# Dropdown symbols
# -------------------------
symbols_map = {
    # Indices
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    # Stocks (F&O list examples)
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "SBIN": "SBIN.NS",
    "INFY": "INFY.NS",
    "HDFC": "HDFCBANK.NS"
}

st.title("📊 Intraday Options Dashboard (Indices + Stocks)")

col1, col2 = st.columns([1, 1])
with col1:
    symbol_choice = st.selectbox("Select Symbol", list(symbols_map.keys()), index=0)
with col2:
    refresh_sec = st.slider("Auto-refresh (sec)", min_value=30, max_value=900, value=300, step=30)

if st.button("Refresh Now"):
    st.experimental_rerun()

# -------------------------
# Helper: round ATM
# -------------------------
def round_strike(spot, sym):
    if sym == "NIFTY":
        step = 50
    elif sym == "BANKNIFTY":
        step = 100
    else:
        step = 50  # most stocks have 50 strike step
    return int(round(spot / step) * step)

# -------------------------
# Fetch spot data
# -------------------------
yfsymbol = symbols_map[symbol_choice]
with st.spinner(f"Fetching price data for {symbol_choice}..."):
    try:
        price_df = yf.Ticker(yfsymbol).history(period="5d", interval="5m").reset_index().dropna()
    except Exception as e:
        st.error(f"Price fetch failed: {e}")
        price_df = pd.DataFrame()

if price_df.empty:
    st.error("No price data available.")
    st.stop()

spot = price_df["Close"].iloc[-1]
atm = round_strike(spot, symbol_choice)

col_a, col_b = st.columns(2)
col_a.metric("Spot", f"{spot:.2f}")
col_b.metric("ATM Strike", atm)

# -------------------------
# Price Chart
# -------------------------
st.subheader(f"{symbol_choice} Price Chart (5m)")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=price_df["Datetime"],
    open=price_df["Open"], high=price_df["High"],
    low=price_df["Low"], close=price_df["Close"],
    name="Price"
))
fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Option Chain
# -------------------------
st.subheader(f"{symbol_choice} Option Chain (ATM ± 300)")
try:
    oc = nse_optionchain_scrapper(symbol_choice)
except Exception as e:
    st.error(f"Option-chain fetch failed: {e}")
    oc = None

if oc:
    records = oc.get("records", {}).get("data", [])
    expiry = oc.get("records", {}).get("expiryDates", [None])[0]

    rows = []
    for d in records:
        try:
            if expiry and d.get("expiryDate") != expiry:
                continue
            strike = d.get("strikePrice")
            ce = d.get("CE") or {}
            pe = d.get("PE") or {}
            rows.append({
                "Strike": strike,
                "CE_OI": ce.get("openInterest", 0),
                "PE_OI": pe.get("openInterest", 0),
                "CE_Chng_OI": ce.get("changeinOpenInterest", 0),
                "PE_Chng_OI": pe.get("changeinOpenInterest", 0),
            })
        except Exception:
            continue

    df_oc = pd.DataFrame(rows).drop_duplicates(subset="Strike").sort_values("Strike")
    df_filtered = df_oc[(df_oc["Strike"] >= atm - 300) & (df_oc["Strike"] <= atm + 300)].copy()

    df_filtered["PCR_OI"] = (df_filtered["PE_OI"] / df_filtered["CE_OI"]).replace([np.inf, -np.inf], np.nan).round(2)

    st.dataframe(df_filtered.reset_index(drop=True).style.format({
        "CE_OI": "{:,}",
        "PE_OI": "{:,}",
        "CE_Chng_OI": "{:+,}",
        "PE_Chng_OI": "{:+,}",
        "PCR_OI": "{:.2f}"
    }))
