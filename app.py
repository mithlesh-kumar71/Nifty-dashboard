import streamlit as st
import pandas as pd
import datetime
import time
from nsepython import option_chain
import yfinance as yf

# Title
st.title("Live NIFTY Option Chain with PCR (ATM ± 500)")

# Get current NIFTY spot price
def get_nifty_spot():
    ticker = yf.Ticker("^NSEI")
    data = ticker.history(period="1d", interval="1m")
    if not data.empty:
        return round(data["Close"].iloc[-1], 2)
    return None

# Fetch Option Chain
def get_option_chain(symbol="NIFTY"):
    try:
        data = option_chain(symbol)
        records = data['records']['data']
        oc_rows = []
        for r in records:
            ce_data = r.get("CE", None)
            pe_data = r.get("PE", None)
            if ce_data and pe_data:
                oc_rows.append({
                    "Strike Price": r["strikePrice"],
                    "CE_OI": ce_data.get("openInterest", 0),
                    "CE_Chng_OI": ce_data.get("changeinOpenInterest", 0),
                    "PE_OI": pe_data.get("openInterest", 0),
                    "PE_Chng_OI": pe_data.get("changeinOpenInterest", 0)
                })
        df = pd.DataFrame(oc_rows)
        return df
    except Exception as e:
        st.error(f"Failed to fetch Option Chain: {e}")
        return pd.DataFrame()

# Filter ATM ± 500 points
def filter_atm_range(df, spot_price):
    lower = spot_price - 500
    upper = spot_price + 500
    return df[(df["Strike Price"] >= lower) & (df["Strike Price"] <= upper)]

# Calculate PCR
def calculate_pcr(df):
    total_ce = df["CE_OI"].sum()
    total_pe = df["PE_OI"].sum()
    return round(total_pe / total_ce, 2) if total_ce > 0 else None

# Save to Excel
def save_to_excel(df, filename="nifty_option_chain.xlsx"):
    try:
        with pd.ExcelWriter(filename, engine="openpyxl", mode="w") as writer:
            df.to_excel(writer, sheet_name="OptionChain", index=False)
        st.success(f"Saved live data to {filename}")
    except Exception as e:
        st.error(f"Excel save error: {e}")

# Main Execution
spot_price = get_nifty_spot()
if spot_price:
    st.write(f"### Current NIFTY Spot Price: {spot_price}")

    df_oc = get_option_chain("NIFTY")
    if not df_oc.empty:
        df_filtered = filter_atm_range(df_oc, spot_price)
        pcr = calculate_pcr(df_filtered)

        st.write("### Option Chain Data (ATM ± 500)")
        st.dataframe(df_filtered)

        st.write(f"### Put-Call Ratio (PCR): {pcr}")

        # Save filtered data to Excel
        save_to_excel(df_filtered)

        # Auto-refresh every 15 mins
        st_autorefresh = st.button("Refresh Now")
        if st_autorefresh:
            st.experimental_rerun()
else:
    st.error("Unable to fetch NIFTY spot price.")
