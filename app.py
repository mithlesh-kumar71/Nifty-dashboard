# NSE Options Live Dashboard (Streamlit)
# Features: India VIX, PCR, ATM Straddle, Bid-Ask spreads, ATM IV, RSI & ADX on spot (NIFTY/BANKNIFTY)
# Notes:
# - This app pulls public NSE option-chain data and Yahoo Finance intraday candles.
# - Use responsibly and respect NSE terms. Educational use only.

import time
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objs as go
import yfinance as yf
import pytz

# ------------------------------
# Utility: Black–Scholes + IV
# ------------------------------

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(S, K, T, r, q, sigma, option_type="C"):
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if option_type == "C" else (K - S))
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "C":
        return S * math.exp(-q*T) * _norm_cdf(d1) - K * math.exp(-r*T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r*T) * _norm_cdf(-d2) - S * math.exp(-q*T) * _norm_cdf(-d1)

def implied_volatility(target_price, S, K, T, r=0.0, q=0.0, option_type="C", tol=1e-6, max_iter=100):
    if target_price is None or target_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan
    low, high = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        price = bs_price(S, K, T, r, q, mid, option_type)
        if abs(price - target_price) < tol:
            return mid
        if price > target_price:
            high = mid
        else:
            low = mid
    return np.nan

# ------------------------------
# Technicals: RSI & ADX
# ------------------------------

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_adx(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(period).sum() / atr)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx = dx.rolling(period).mean()
    return plus_di, minus_di, adx

# ------------------------------
# NSE fetchers
# ------------------------------

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com/",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

def nse_session():
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    s.get("https://www.nseindia.com", timeout=10)
    return s

def fetch_option_chain(symbol="NIFTY"):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    s = nse_session()
    resp = s.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()

def parse_option_chain(json_data):
    records = json_data.get("records", {})
    underlying = records.get("underlyingValue")
    expiry_list = records.get("expiryDates", [])
    data = records.get("data", [])
    df_rows = []
    for row in data:
        strike = row.get("strikePrice")
        expiry = row.get("expiryDate")
        ce = row.get("CE", {})
        pe = row.get("PE", {})
        df_rows.append({
            "expiry": expiry, "strike": strike,
            "CE_bid": ce.get("bidprice"), "CE_ask": ce.get("askPrice"),
            "CE_ltp": ce.get("lastPrice"), "CE_oi": ce.get("openInterest"),
            "CE_tottrdqty": ce.get("totalTradedVolume"),
            "PE_bid": pe.get("bidprice"), "PE_ask": pe.get("askPrice"),
            "PE_ltp": pe.get("lastPrice"), "PE_oi": pe.get("openInterest"),
            "PE_tottrdqty": pe.get("totalTradedVolume")
        })
    df = pd.DataFrame(df_rows)
    return underlying, expiry_list, df

def compute_pcr_from_chain(df, expiry=None):
    dfe = df[df["expiry"] == expiry] if expiry else df
    call_oi = dfe["CE_oi"].fillna(0).sum()
    put_oi  = dfe["PE_oi"].fillna(0).sum()
    if call_oi == 0:
        return np.nan
    return put_oi / call_oi

def nearest_atm_strike(spot, strikes):
    return min(strikes, key=lambda k: abs(k - spot))

def compute_atm_straddle(df, spot, expiry):
    dfe = df[df["expiry"] == expiry]
    if dfe.empty:
        return np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan), (np.nan, np.nan)
    strike = nearest_atm_strike(spot, dfe["strike"].dropna().unique().tolist())
    row = dfe[dfe["strike"] == strike].iloc[0]
    ce = row.get("CE_ltp", np.nan)
    pe = row.get("PE_ltp", np.nan)
    ce_ba = (row.get("CE_bid", np.nan), row.get("CE_ask", np.nan))
    pe_ba = (row.get("PE_bid", np.nan), row.get("PE_ask", np.nan))
    total = (ce or 0) + (pe or 0)
    return strike, ce, pe, total, ce_ba, pe_ba

def derive_atm_iv(spot, strike, expiry_dt, ce_ltp, pe_ltp):
    T = max((expiry_dt - datetime.now(timezone.utc)).total_seconds() / (365*24*3600), 1e-6)
    if np.isnan(T) or T <= 0:
        return np.nan, np.nan
    ce_iv = implied_volatility(ce_ltp, spot, strike, T, option_type="C") if pd.notna(ce_ltp) else np.nan
    pe_iv = implied_volatility(pe_ltp, spot, strike, T, option_type="P") if pd.notna(pe_ltp) else np.nan
    return ce_iv, pe_iv

def nse_index_spot(symbol="NIFTY"):
    yf_symbol = "^NSEI" if symbol == "NIFTY" else "^NSEBANK" if symbol == "BANKNIFTY" else None
    if yf_symbol is None:
        return None
    try:
        tkr = yf.Ticker(yf_symbol)
        price = tkr.history(period="1d", interval="1m")["Close"].iloc[-1]
        return float(price)
    except Exception:
        return None

def fetch_india_vix_series():
    try:
        vix = yf.Ticker("^INDIAVIX").history(period="5d", interval="5m")["Close"]
        return vix
    except Exception:
        return None

# ------------------------------
# Streamlit App
# ------------------------------

st.set_page_config(page_title="NSE Options Live Dashboard", layout="wide")

st.title("📊 NSE Options Live Dashboard")
st.caption("India VIX • PCR • ATM Straddle • Bid–Ask • ATM IV • RSI & ADX (NIFTY/BANKNIFTY)")

with st.sidebar:
    st.header("Controls")
    symbol = st.selectbox("Index", ["NIFTY", "BANKNIFTY"], index=0)
    refresh_sec = st.slider("Auto-refresh (seconds)", min_value=15, max_value=180, value=60, step=15)
    tech_period = st.number_input("RSI/ADX Period", min_value=7, max_value=30, value=14, step=1)
    st.write(" ")
    st.info("Tip: Keep refresh >= 30s to avoid throttling.")

# --- Fetch option chain
err = None
try:
    oc_json = fetch_option_chain(symbol)
    spot, expiries, oc_df = parse_option_chain(oc_json)
except Exception as e:
    oc_json, spot, expiries, oc_df = None, None, [], pd.DataFrame()
    err = str(e)

if spot is None:
    spot = nse_index_spot(symbol)

col1, col2, col3, col4, col5, col6 = st.columns(6)

vix_series = fetch_india_vix_series()
vix_last = vix_series.iloc[-1] if vix_series is not None and len(vix_series) else np.nan

expiry_choice = expiries[0] if expiries else None

pcr = compute_pcr_from_chain(oc_df, expiry_choice) if not oc_df.empty else np.nan

if not oc_df.empty and spot is not None and expiry_choice:
    strike, ce, pe, straddle, ce_ba, pe_ba = compute_atm_straddle(oc_df, spot, expiry_choice)
    try:
        expiry_dt_naive = datetime.strptime(expiry_choice, "%d-%b-%Y")
        expiry_dt_ist = expiry_dt_naive.replace(hour=15, minute=30)
        ist = pytz.timezone("Asia/Kolkata")
        expiry_dt_ist = ist.localize(expiry_dt_ist)
        expiry_dt_utc = expiry_dt_ist.astimezone(timezone.utc)
    except Exception:
        expiry_dt_utc = datetime.now(timezone.utc)
    ce_iv, pe_iv = derive_atm_iv(spot, strike, expiry_dt_utc, ce, pe)
    bidask_spread_ce = (ce_ba[1] - ce_ba[0]) if all(pd.notna(x) for x in ce_ba) else np.nan
    bidask_spread_pe = (pe_ba[1] - pe_ba[0]) if all(pd.notna(x) for x in pe_ba) else np.nan
else:
    strike = ce = pe = straddle = ce_iv = pe_iv = bidask_spread_ce = bidask_spread_pe = np.nan

yf_symbol = "^NSEI" if symbol == "NIFTY" else "^NSEBANK"
try:
    hist = yf.Ticker(yf_symbol).history(period="2d", interval="1m").dropna()
    rsi = compute_rsi(hist["Close"], period=tech_period)
    plus_di, minus_di, adx = compute_adx(hist, period=tech_period)
except Exception:
    hist = None
    rsi = plus_di = minus_di = adx = None

col1.metric("Spot", f"{spot:,.2f}" if pd.notna(spot) else "—")
col2.metric("India VIX (last)", f"{vix_last:,.2f}" if pd.notna(vix_last) else "—")
col3.metric("PCR (OI)", f"{pcr:,.2f}" if pd.notna(pcr) else "—")
col4.metric("ATM Straddle", f"{straddle:,.2f}" if pd.notna(straddle) else "—")
col5.metric("ATM IV (CE)", f"{ce_iv*100:,.2f}%" if pd.notna(ce_iv) else "—")
col6.metric("ATM IV (PE)", f"{pe_iv*100:,.2f}%" if pd.notna(pe_iv) else "—")

st.divider()

left, right = st.columns([1,1])
with left:
    st.subheader("India VIX (last 5 days, 5m)")
    if vix_series is not None and len(vix_series):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vix_series.index, y=vix_series.values, mode="lines", name="VIX"))
        fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("VIX series unavailable.")

with right:
    st.subheader(f"{symbol} RSI & ADX (1m)")
    if hist is not None and rsi is not None and adx is not None:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hist.index, y=rsi, mode="lines", name="RSI"))
        fig2.add_trace(go.Scatter(x=hist.index, y=adx, mode="lines", name="ADX"))
        fig2.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=300)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Intraday data unavailable for RSI/ADX.")

st.subheader("ATM Snapshot")
if pd.notna(strike):
    st.write(f"**ATM Strike:** {int(strike)}  |  **CE Bid–Ask:** {ce_ba}  (Spread: {bidask_spread_ce})  |  **PE Bid–Ask:** {pe_ba}  (Spread: {bidask_spread_pe})")
else:
    st.write("No ATM data.")

st.subheader(f"Option Chain (Top strikes near ATM) — {expiry_choice or ''}")
if not oc_df.empty and pd.notna(spot):
    strikes = sorted(oc_df["strike"].dropna().unique())
    if strikes:
        atm = nearest_atm_strike(spot, strikes)
        window = 6
        step = strikes[1]-strikes[0] if len(strikes)>1 else 50
        near = [k for k in strikes if abs(k - atm) <= window * step]
        view = oc_df[(oc_df["expiry"]==expiry_choice) & (oc_df["strike"].isin(near))].copy()
        view = view.sort_values("strike")
        cols = ["strike","CE_bid","CE_ask","CE_ltp","CE_oi","PE_bid","PE_ask","PE_ltp","PE_oi"]
        st.dataframe(view[cols].reset_index(drop=True), use_container_width=True)
else:
    st.info("Option chain unavailable.")

st.caption("⚠️ Data is fetched from public sources (NSE / Yahoo) and may be delayed or rate-limited. Do not use for automated trading.")

# --- Auto-refresh fallback (JS reload)
st.markdown(f"""
<script>
setTimeout(function(){{
    window.location.reload();
}}, {int(refresh_sec)*1000});
</script>
""", unsafe_allow_html=True)
