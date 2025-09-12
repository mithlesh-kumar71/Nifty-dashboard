import streamlit as st
import pandas as pd
import datetime
import time
from nsepython import option_chain
import yfinance as yf

# Title
st.title("Live NIFTY Option Chain with PCR (ATM ± 300)")

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
                ce_chg_oi = ce_data.get("changeinOpenInterest", 0)
                pe_chg_oi = pe_data.get("changeinOpenInterest", 0)
                ratio = round(pe_chg_oi / ce_chg_oi, 2) if ce_chg_oi != 0 else None

                oc_rows.append({
                    "Strike Price": r["strikePrice"],
                    "CE_OI": ce_data.get("openInterest", 0),
                    "CE_Chng_OI": ce_chg_oi,
                    "PE_OI": pe_data.get("openInterest", 0),
                    "PE_Chng_OI": pe_chg_oi,
                    "PE/CE_Chng_OI_Ratio": ratio
                })
        df = pd.DataFrame(oc_rows)
        return df
    except Exception as e:
        st.error(f"Failed to fetch Option Chain: {e}")
        return pd.DataFrame()

# Filter ATM ± 300 points
def filter_atm_range(df, spot_price, buffer=300):
    lower = spot_price - buffer
    upper = spot_price + buffer
    return df[(df["Strike Price"] >= lower) & (df["Strike Price"] <= upper)]

# Calculate PCR
def calculate_pcr(df):
    total_ce = df["CE_OI"].sum()
    total_pe = df["PE_OI"].sum()
    return round(total_pe / total_ce, 2) if total_ce > 0 else None

# Highlight ATM Strike
def highlight_atm(row, atm_strike):
    return ['background-color: yellow' if row["Strike Price"] == atm_strike else '' for _ in row]

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
        df_filtered = filter_atm_range(df_oc, spot_price, buffer=300)

        # Find nearest ATM strike
        atm_strike = min(df_filtered["Strike Price"], key=lambda x: abs(x - spot_price))

        pcr = calculate_pcr(df_filtered)

        st.write("### Option Chain Data (ATM ± 300)")
        st.dataframe(df_filtered.style.apply(highlight_atm, atm_strike=atm_strike, axis=1))

        st.write(f"### Put-Call Ratio (PCR): {pcr}")

        # Save filtered data to Excel
        save_to_excel(df_filtered)

        # Refresh button
        if st.button("Refresh Now"):
            st.experimental_rerun()
else:
    st.error("Unable to fetch NIFTY spot price.")
