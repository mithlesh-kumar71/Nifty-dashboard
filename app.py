import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from ta.volatility import AverageTrueRange

# --------- Streamlit page config ----------
st.set_page_config(page_title="Options Dashboard", layout="wide")

# --------- Watchlist ----------
WATCHLIST = ["^NSEI", "RELIANCE.NS", "TCS.NS", "KOTAKBANK.NS", "SBIN.NS", 
             "TATAMOTORS.NS", "HDFC.NS", "BEL.NS", "BAJFINANCE.NS", "HAL.NS", 
             "INFY.NS", "VEDL.NS"]

st.title("Intraday Options Dashboard")

# ---------- Stock selection ----------
symbol = st.selectbox("Select Stock/Index", WATCHLIST)

# ---------- Time interval ----------
interval = st.selectbox("Select Interval", ["15m", "30m", "1h", "1d"])
period = "30d" if interval in ["15m", "30m", "1h"] else "1y"

# ---------- Fetch OHLCV ----------
@st.cache_data(ttl=60)
def fetch_ohlcv(ticker, period="30d", interval="15m"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(i) for i in col]).strip() for col in df.columns.values]

    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    existing_cols = [c for c in expected_cols if c in df.columns]
    if not existing_cols:
        return pd.DataFrame()
    df = df.dropna(subset=existing_cols)
    df.reset_index(inplace=True)
    return df

df = fetch_ohlcv(symbol, period=period, interval=interval)

if df.empty:
    st.warning("No data fetched for this stock/interval.")
    st.stop()

# ---------- Supertrend ----------
def supertrend(df, period=10, multiplier=3):
    df = df.copy()
    hl2 = (df["High"] + df["Low"]) / 2
    atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=period).average_true_range()
    df['UpperBand'] = hl2 + multiplier * atr
    df['LowerBand'] = hl2 - multiplier * atr
    df['Trend'] = True  # True = uptrend, False = downtrend
    for i in range(1, len(df)):
        if df['Close'].iat[i-1] <= df['UpperBand'].iat[i-1]:
            df['Trend'].iat[i] = True
        else:
            df['Trend'].iat[i] = False
    return df

df = supertrend(df)

# ---------- Display OHLC + Trend ----------
st.subheader(f"{symbol} Price Data & Trend")
st.dataframe(df[['Datetime','Open','High','Low','Close','Trend']])

# ---------- Option Chain Placeholder ----------
st.subheader("Option Chain & PCR (ATM ± 300)")

st.info("Option Chain fetching not included to avoid NSE scraping errors. In production, replace this with nsepython or API-based data.")

# ---------- Simulated PCR data for ATM strike ----------
atm_strike = round(df["Close"].iloc[-1] / 50) * 50
strike_range = list(range(atm_strike-300, atm_strike+301, 50))
pcr_values = np.random.uniform(0.7, 1.3, len(strike_range))
chng_oi_ratio = np.random.uniform(0.5, 1.5, len(strike_range))

pcr_df = pd.DataFrame({
    "Strike Price": strike_range,
    "PCR": np.round(pcr_values,2),
    "Chng_OI_Ratio": np.round(chng_oi_ratio,2)
})

# Highlight ATM
def highlight_atm(row):
    return ['background-color: yellow' if row["Strike Price"]==atm_strike else '' for _ in row]

st.dataframe(pcr_df.style.apply(highlight_atm, axis=1))

# ---------- Plot PCR vs Change in OI ----------
st.subheader("ATM PCR Change Chart")
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(pcr_df["Strike Price"], pcr_df["Chng_OI_Ratio"], marker='o', label='Chng_OI_Ratio')
ax.axhline(1, color='gray', linestyle='--', label='Neutral=1')
ax.set_xlabel("Strike Price")
ax.set_ylabel("Change in OI PCR")
ax.set_title(f"{symbol} ATM PCR ±300")
ax.legend()
st.pyplot(fig)

st.success(f"ATM Strike: {atm_strike}")

