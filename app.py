import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from nsepython import nse_optionchain_scrapper

# Streamlit Page Config
st.set_page_config(page_title="NSE Option Chain + PCR Dashboard", layout="wide")

st.title("📊 NSE Option Chain & PCR Dashboard")

# Dropdown for symbol
symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "SBIN"])

# Fetch Option Chain Data
@st.cache_data(ttl=300)  # cache for 5 minutes
def fetch_option_chain(symbol):
    try:
        data = nse_optionchain_scrapper(symbol)
        return data
    except Exception as e:
        st.error(f"Error fetching option chain: {e}")
        return None

def calculate_pcr(df):
    """Calculates PCR for OI and Change in OI."""
    df["PCR_OI"] = (df["PE_OI"] / df["CE_OI"]).replace([float("inf"), -float("inf")], 0).fillna(0)
    df["PCR_Chng_OI"] = (df["PE_Chng_OI"] / df["CE_Chng_OI"]).replace([float("inf"), -float("inf")], 0).fillna(0)
    return df

data = fetch_option_chain(symbol)

if data:
    # Convert to DataFrame
    ce_data = pd.DataFrame(data['records']['data'])
    ce_data = ce_data[['strikePrice', 'CE', 'PE']]

    # Extract required fields
    rows = []
    for _, row in ce_data.iterrows():
        strike = row['strikePrice']
        ce_oi = row['CE']['openInterest'] if 'CE' in row and row['CE'] else 0
        pe_oi = row['PE']['openInterest'] if 'PE' in row and row['PE'] else 0
        ce_chng = row['CE']['changeinOpenInterest'] if 'CE' in row and row['CE'] else 0
        pe_chng = row['PE']['changeinOpenInterest'] if 'PE' in row and row['PE'] else 0
        rows.append([strike, ce_oi, pe_oi, ce_chng, pe_chng])

    df = pd.DataFrame(rows, columns=["Strike Price", "CE_OI", "PE_OI", "CE_Chng_OI", "PE_Chng_OI"])

    # Calculate PCR
    df = calculate_pcr(df)

    # Find ATM Strike (nearest to last price from underlying)
    try:
        underlying_price = data['records']['underlyingValue']
        atm_strike = min(df["Strike Price"], key=lambda x: abs(x - underlying_price))
    except:
        atm_strike = df["Strike Price"].median()

    # Filter ±300 points range around ATM
    df_filtered = df[(df["Strike Price"] >= atm_strike - 300) & (df["Strike Price"] <= atm_strike + 300)]

    # Highlight ATM row
    def highlight_atm(row):
        return ['background-color: yellow' if row["Strike Price"] == atm_strike else '' for _ in row]

    st.subheader(f"Option Chain Data (ATM = {atm_strike})")
    st.dataframe(df_filtered.style.apply(highlight_atm, axis=1).format("{:.2f}", subset=["PCR_OI", "PCR_Chng_OI"]))

    # Plot PCR Change in OI (ATM Strike)
    atm_row = df[df["Strike Price"] == atm_strike]
    if not atm_row.empty:
        atm_pcr_chng = atm_row["PCR_Chng_OI"].values[0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered["Strike Price"],
            y=df_filtered["PCR_Chng_OI"],
            mode="lines+markers",
            name="PCR Change OI"
        ))

        # Reference line at 1
        fig.add_hline(y=1, line_dash="dot", line_color="red")

        fig.update_layout(
            title="PCR Change in OI Around ATM",
            xaxis_title="Strike Price",
            yaxis_title="PCR Change OI",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data available. Please try again later.")
