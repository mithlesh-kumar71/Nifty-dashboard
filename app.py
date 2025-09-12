import streamlit as st
import pandas as pd
import datetime
import yfinance as yf
from nsepython import option_chain
import os

# Title
st.title("📊 NIFTY Option Chain + PCR & OI Trend (ATM ± 300)")

# --- Get NIFTY Spot Price ---
def get_nifty_spot():
    ticker = yf.Ticker("^NSEI")
    data = ticker.history(period="1d", interval="1m")
    if not data.empty:
        return round(data["Close"].iloc[-1], 2)
    return None

# --- Fetch Option Chain ---
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

# --- Filter ATM ± 300 points ---
def filter_atm_range(df, spot_price, buffer=300):
    lower = spot_price - buffer
    upper = spot_price + buffer
    return df[(df["Strike Price"] >= lower) & (df["Strike Price"] <= upper)]

# --- PCR Calculation ---
def calculate_pcr(df):
    total_ce = df["CE_OI"].sum()
    total_pe = df["PE_OI"].sum()
    return round(total_pe / total_ce, 2) if total_ce > 0 else None, total_ce, total_pe

# --- Highlight ATM Strike ---
def highlight_atm(row, atm_strike):
    return ['background-color: yellow' if row["Strike Price"] == atm_strike else '' for _ in row]

# --- Generate Current Time Slot ---
def get_current_slot():
    now = datetime.datetime.now().time()
    minutes = (now.minute // 15) * 15
    slot = datetime.time(now.hour, minutes)
    return slot.strftime("%H:%M")

# --- Save PCR + OI Trend to Excel ---
def save_trend(time_slot, pcr, total_ce, total_pe, filename="pcr_oi_trend.xlsx"):
    new_row = pd.DataFrame([[time_slot, pcr, total_ce, total_pe]],
                           columns=["Time Slot", "PCR", "Total CE OI", "Total PE OI"])

    if os.path.exists(filename):
        old = pd.read_excel(filename)
        df = pd.concat([old, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_excel(filename, index=False)
    return df

# --- Main ---
spot_price = get_nifty_spot()
if spot_price:
    st.write(f"### Current NIFTY Spot Price: {spot_price}")

    df_oc = get_option_chain("NIFTY")
    if not df_oc.empty:
        df_filtered = filter_atm_range(df_oc, spot_price, buffer=300)

        # ATM Strike
        atm_strike = min(df_filtered["Strike Price"], key=lambda x: abs(x - spot_price))

        # PCR + OI
        pcr, total_ce, total_pe = calculate_pcr(df_filtered)
        time_slot = get_current_slot()

        st.write(f"### Current PCR: {pcr} | CE OI: {total_ce:,} | PE OI: {total_pe:,} at {time_slot}")

        # Display Option Chain
        st.dataframe(df_filtered.style.apply(highlight_atm, atm_strike=atm_strike, axis=1))

        # Save Trend Data
        df_trend = save_trend(time_slot, pcr, total_ce, total_pe)

        # PCR Trend
        st.subheader("📈 PCR Trend (Intraday)")
        st.line_chart(df_trend.set_index("Time Slot")["PCR"])

        # OI Trend
        st.subheader("📊 OI Build-up Trend (PE vs CE)")
        st.line_chart(df_trend.set_index("Time Slot")[["Total CE OI", "Total PE OI"]])
else:
    st.error("Unable to fetch NIFTY spot price.")
