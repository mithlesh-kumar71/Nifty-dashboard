import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from nsepython import nse_optionchain_scrapper
from streamlit_autorefresh import st_autorefresh

# -------------------------
# Page config (first Streamlit command)
# -------------------------
st.set_page_config(page_title="Intraday Strategy + Options Dashboard", layout="wide")

# -------------------------
# Auto-refresh every 15 minutes (900000 ms)
# -------------------------
st_autorefresh(interval=900000, key="refresh")

# -------------------------
# App Title
# -------------------------
st.title("📊 NSE Intraday Strategy + Options Dashboard")

# -------------------------
# Step 1: Fetch ATM Strike from Yahoo Finance
# -------------------------
ticker = "^NSEI"  # Nifty 50 Index
nifty_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
atm_strike = round(nifty_price / 50) * 50
st.subheader(f"📍 Nifty Spot: {nifty_price:.2f} | ATM Strike: {atm_strike}")

# -------------------------
# Step 2: Fetch Option Chain Data
# -------------------------
symbol = "NIFTY"
try:
    oc_data = nse_optionchain_scrapper(symbol)   # ✅ only one argument

    records = []
    for d in oc_data['records']['data']:
        strike = d['strikePrice']
        ce_oi = d['CE']['openInterest'] if 'CE' in d else 0
        ce_chng_oi = d['CE']['changeinOpenInterest'] if 'CE' in d else 0
        pe_oi = d['PE']['openInterest'] if 'PE' in d else 0
        pe_chng_oi = d['PE']['changeinOpenInterest'] if 'PE' in d else 0

        records.append({
            "Strike Price": strike,
            "CE OI": ce_oi,
            "CE Chng OI": ce_chng_oi,
            "PE OI": pe_oi,
            "PE Chng OI": pe_chng_oi,
            "PCR (OI)": round(pe_oi / ce_oi, 2) if ce_oi != 0 else np.nan,
            "PCR (Chng OI)": round(pe_chng_oi / ce_chng_oi, 2) if ce_chng_oi != 0 else np.nan
        })

    df = pd.DataFrame(records)

    # -------------------------
    # Step 3: Filter for ATM ± 300
    # -------------------------
    df_filtered = df[(df["Strike Price"] >= atm_strike - 300) & (df["Strike Price"] <= atm_strike + 300)]

    # Highlight ATM Strike
    def highlight_atm(row):
        color = 'background-color: yellow' if row["Strike Price"] == atm_strike else ''
        return [color] * len(row)

    st.subheader("📑 Option Chain Data (ATM ± 300)")
    st.dataframe(df_filtered.style.apply(highlight_atm, axis=1))

    # -------------------------
    # Step 4: Plot Line Chart for Change in OI PCR (ATM only)
    # -------------------------
    atm_row = df_filtered[df_filtered["Strike Price"] == atm_strike]

    if not atm_row.empty:
        latest_ratio = atm_row["PCR (Chng OI)"].iloc[0]

        if "pcr_history" not in st.session_state:
            st.session_state["pcr_history"] = []

        st.session_state["pcr_history"].append({
            "time": datetime.now().strftime("%H:%M"),
            "PCR (Chng OI)": latest_ratio
        })

        pcr_df = pd.DataFrame(st.session_state["pcr_history"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pcr_df["time"], y=pcr_df["PCR (Chng OI)"], mode="lines+markers",
            name="PCR (Chng OI)", line=dict(color="blue")
        ))
        fig.add_hline(y=1, line_dash="dot", line_color="red")  # Neutral Line at 1

        fig.update_layout(
            title="ATM Strike PCR (Change in OI) Over Time",
            xaxis_title="Time",
            yaxis_title="PCR (Chng OI)",
            yaxis=dict(range=[0, max(2, pcr_df["PCR (Chng OI)"].max() + 0.5)])
        )

        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error fetching data: {e}")
