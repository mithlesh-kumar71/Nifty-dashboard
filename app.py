import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from nsepython import nse_optionchain_scrapper, nse_index_quote

# ------------------- Helper Function -------------------
def fetch_option_chain(symbol="NIFTY"):
    try:
        data = nse_optionchain_scrapper(symbol)
        records = []
        for rec in data["records"]["data"]:
            if "CE" in rec and "PE" in rec:
                records.append({
                    "strikePrice": rec["strikePrice"],
                    "CE_OI": rec["CE"]["openInterest"],
                    "CE_ChngOI": rec["CE"]["changeinOpenInterest"],
                    "PE_OI": rec["PE"]["openInterest"],
                    "PE_ChngOI": rec["PE"]["changeinOpenInterest"],
                })
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"Error fetching option chain: {e}")
        return pd.DataFrame()

# ------------------- Main -------------------
st.set_page_config(page_title="Options Dashboard", layout="wide")
st.title("📊 NIFTY Options Dashboard")

symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "SBIN"])
df = fetch_option_chain(symbol)

if not df.empty:
    # ATM Calculation
    spot = nse_index_quote(symbol)["lastPrice"]
    atm_strike = min(df["strikePrice"], key=lambda x: abs(x - spot))

    # Filter ±300 points around ATM
    df_filtered = df[(df["strikePrice"] >= atm_strike - 300) & (df["strikePrice"] <= atm_strike + 300)]

    # Add PCR columns
    df_filtered["PCR_OI"] = (df_filtered["PE_OI"] / df_filtered["CE_OI"]).round(2)
    df_filtered["PCR_ChngOI"] = (df_filtered["PE_ChngOI"] / df_filtered["CE_ChngOI"]).round(2)

    st.subheader(f"Option Chain (±300 points of ATM {atm_strike})")
    st.dataframe(df_filtered.style.apply(
        lambda row: ['background-color: yellow' if row["strikePrice"] == atm_strike else '' for _ in row],
        axis=1
    ))

    # ------------------- Plotly Line Chart -------------------
    st.subheader("📈 PCR Change in OI (ATM Strike)")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_filtered["strikePrice"],
        y=df_filtered["PCR_ChngOI"],
        mode="lines+markers",
        name="Change in OI PCR",
        line=dict(color="blue"),
    ))

    # Reference line at 1
    fig.add_hline(y=1, line_dash="dash", line_color="red")

    fig.update_layout(
        title=f"Change in OI PCR around ATM {atm_strike}",
        xaxis_title="Strike Price",
        yaxis_title="PCR (PE_ChngOI / CE_ChngOI)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No option chain data available right now.")
