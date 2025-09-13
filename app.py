import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nsepython import nse_optionchain_scrapper

# -----------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------
st.set_page_config(page_title="Intraday Options Dashboard", layout="wide")

st.title("📊 Intraday Options Trading Dashboard")

# -----------------------------------------------------
# Dropdown for index / stocks
# -----------------------------------------------------
symbol = st.selectbox(
    "Select Symbol",
    ["NIFTY", "RELIANCE", "TCS", "SBIN"]
)

# -----------------------------------------------------
# Fetch Option Chain Data
# -----------------------------------------------------
try:
    option_chain = nse_optionchain_scrapper(symbol)
    records = option_chain['records']['data']

    # Flatten CE and PE legs
    data = []
    for r in records:
        strike = r['strikePrice']
        ce_oi = r['CE']['openInterest'] if 'CE' in r else 0
        ce_chng_oi = r['CE']['changeinOpenInterest'] if 'CE' in r else 0
        pe_oi = r['PE']['openInterest'] if 'PE' in r else 0
        pe_chng_oi = r['PE']['changeinOpenInterest'] if 'PE' in r else 0

        ratio = round((pe_chng_oi / ce_chng_oi), 2) if ce_chng_oi != 0 else np.nan

        data.append({
            "Strike Price": strike,
            "CE_OI": ce_oi,
            "PE_OI": pe_oi,
            "CE_Chng_in_OI": ce_chng_oi,
            "PE_Chng_in_OI": pe_chng_oi,
            "PE/CE_Chng_OI_Ratio": ratio
        })

    df = pd.DataFrame(data)

except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# -----------------------------------------------------
# Filter Data: ATM ± 300
# -----------------------------------------------------
underlying = option_chain['records']['underlyingValue']
atm_strike = int(round(underlying / 50) * 50)   # nearest 50
df_filtered = df[(df["Strike Price"] >= atm_strike - 300) & 
                 (df["Strike Price"] <= atm_strike + 300)]

# -----------------------------------------------------
# Highlight ATM
# -----------------------------------------------------
def highlight_atm(row):
    return ['background-color: yellow' if row["Strike Price"] == atm_strike else '' for _ in row]

# -----------------------------------------------------
# Display Data
# -----------------------------------------------------
st.subheader(f"Option Chain for {symbol} (ATM Strike: {atm_strike})")

st.dataframe(
    df_filtered.rename(columns={
        "Strike Price": "Strike",
        "CE_OI": "CE OI",
        "PE_OI": "PE OI",
        "CE_Chng_in_OI": "CE Chng OI",
        "PE_Chng_in_OI": "PE Chng OI",
        "PE/CE_Chng_OI_Ratio": "PE/CE OI Ratio"
    }).style.apply(highlight_atm, axis=1)
)

# -----------------------------------------------------
# Plot PCR (ATM Change in OI Ratio)
# -----------------------------------------------------
st.subheader("ATM PCR Trend (Change in OI Ratio)")

atm_row = df[df["Strike Price"] == atm_strike]
if not atm_row.empty:
    latest_ratio = atm_row["PE/CE_Chng_OI_Ratio"].values[0]

    fig, ax = plt.subplots()
    ax.axhline(1, color="black", linestyle="--")  # Neutral line at 1
    ax.plot([0, 1], [latest_ratio, latest_ratio], marker="o",
            color="green" if latest_ratio > 1 else "red")

    ax.set_ylim(0, max(2, latest_ratio + 0.5))
    ax.set_xticks([])
    ax.set_ylabel("PCR (PE/CE Change OI Ratio)")
    ax.set_title(f"ATM Strike {atm_strike} PCR = {latest_ratio:.2f}")

    st.pyplot(fig)
else:
    st.warning("ATM strike not found in option chain data.")
