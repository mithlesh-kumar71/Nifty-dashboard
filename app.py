import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nsepython import nse_optionchain_scrapper

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="NIFTY Options Dashboard", layout="wide")
st.title("📊 NIFTY Options Dashboard with PCR & OI Analysis")

# -------------------------
# Sidebar Controls
# -------------------------
symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "SBIN"], index=0)

# -------------------------
# Fetch Option Chain Data
# -------------------------
@st.cache_data(ttl=60)
def fetch_option_chain(sym):
    try:
        data = nse_optionchain_scrapper(sym)
        ce_data = pd.DataFrame(data['records']['data'])
        return ce_data, data['records']['underlyingValue']
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None

ce_data, underlying = fetch_option_chain(symbol)

if ce_data is None:
    st.stop()

# -------------------------
# Extract OI Data
# -------------------------
rows = []
for _, row in ce_data.iterrows():
    strike = row['strikePrice']

    ce_oi = row['CE']['openInterest'] if isinstance(row['CE'], dict) else 0
    pe_oi = row['PE']['openInterest'] if isinstance(row['PE'], dict) else 0

    ce_chng = row['CE']['changeinOpenInterest'] if isinstance(row['CE'], dict) else 0
    pe_chng = row['PE']['changeinOpenInterest'] if isinstance(row['PE'], dict) else 0

    rows.append([strike, ce_oi, pe_oi, ce_chng, pe_chng])

df = pd.DataFrame(rows, columns=["strike", "CE_OI", "PE_OI", "CE_Chng_OI", "PE_Chng_OI"])

# -------------------------
# Calculate Ratios
# -------------------------
df["PCR"] = (df["PE_OI"] / df["CE_OI"].replace(0, np.nan)).round(2)
df["Chng_OI_PCR"] = (df["PE_Chng_OI"] / df["CE_Chng_OI"].replace(0, np.nan)).round(2)

# -------------------------
# Filter ATM ± 300
# -------------------------
atm_strike = round(underlying / 50) * 50  # rounded to nearest 50
df_filtered = df[(df["strike"] >= atm_strike - 300) & (df["strike"] <= atm_strike + 300)]

# Highlight ATM strike
def highlight_atm(row):
    return ['background-color: yellow' if row["strike"] == atm_strike else '' for _ in row]

st.subheader(f"Underlying: {underlying:.2f} | ATM Strike: {atm_strike}")
st.dataframe(
    df_filtered.style.apply(highlight_atm, axis=1),
    use_container_width=True
)

# -------------------------
# PCR Line Chart (ATM only)
# -------------------------
atm_row = df[df["strike"] == atm_strike]
if not atm_row.empty:
    atm_pcr_series = atm_row["Chng_OI_PCR"].values

    fig, ax = plt.subplots()
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.plot([0], atm_pcr_series, marker="o", markersize=8,
            color="green" if atm_pcr_series[0] > 1 else "red")

    ax.set_ylim(0, max(2, atm_pcr_series[0] + 0.5))
    ax.set_title(f"ATM ({atm_strike}) Change in OI PCR")
    st.pyplot(fig)
else:
    st.warning("ATM strike data not available yet.")
