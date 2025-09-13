# app.py
# Intraday Options Dashboard — Option Chain + IV + Max Pain + Clean PCR
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math
import os

# nsepython helper
from nsepython import nse_optionchain_scrapper

# -------------------------
# Page config (must be first Streamlit Streamlit command)
# -------------------------
st.set_page_config(page_title="Options: PCR • IV • MaxPain", layout="wide")
st.title("📈 Options Dashboard — PCR • IV (approx) • Max Pain")

# -------------------------
# Sidebar: symbol, buffer, refresh controls
# -------------------------
symbols_map = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "SBIN": "SBIN.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
}

symbol_choice = st.sidebar.selectbox("Symbol (index / stock)", list(symbols_map.keys()), index=0)
buffer = st.sidebar.number_input("ATM buffer (points)", min_value=100, max_value=1000, step=50, value=300)
top_n = st.sidebar.number_input("Top movers (N)", min_value=3, max_value=20, value=8)
auto_append = st.sidebar.checkbox("Append PCR to intraday trend (session)", value=True)

if st.sidebar.button("Refresh now"):
    st.experimental_rerun()

# -------------------------
# Helpers: BS price & implied vol
# -------------------------
def bs_price(S, K, T, r, sigma, option_type="C"):
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if option_type == "C" else (K - S))
    from math import log, sqrt, exp
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    from math import erf, sqrt as _sqrt
    N = lambda x: 0.5 * (1.0 + erf(x / _sqrt(2.0)))
    if option_type == "C":
        return S * N(d1) - K * math.exp(-r * T) * N(d2)
    else:
        return K * math.exp(-r * T) * N(-d2) - S * N(-d1)

def implied_vol(target_price, S, K, T, r=0.0, option_type="C"):
    # simple bisection method, returns nan when impossible
    if target_price is None or target_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan
    lo, hi = 1e-6, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p = bs_price(S, K, T, r, mid, option_type)
        if abs(p - target_price) < 1e-6:
            return mid
        if p > target_price:
            hi = mid
        else:
            lo = mid
    return mid

# -------------------------
# Utility: round ATM strike
# -------------------------
def round_strike(spot, symbol):
    if symbol == "NIFTY":
        step = 50
    elif symbol == "BANKNIFTY":
        step = 100
    else:
        # many stocks trade strikes at 50 - use 50 as default
        step = 50
    return int(round(spot / step) * step)

# -------------------------
# Fetch spot price (Yahoo)
# -------------------------
yf_sym = symbols_map[symbol_choice]
with st.spinner("Fetching spot price..."):
    try:
        # fetch last 1 day 1m to get latest close
        spot_df = yf.Ticker(yf_sym).history(period="1d", interval="1m")
        spot = float(spot_df["Close"].dropna().iloc[-1])
    except Exception as e:
        st.error(f"Failed to fetch spot price: {e}")
        spot = None

if spot is None:
    st.stop()

atm_strike = round_strike(spot, symbol_choice)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Symbol", symbol_choice)
col2.metric("Spot", f"{spot:,.2f}")
col3.metric("ATM (rounded)", f"{atm_strike}")
col4.write("")  # placeholder for PCR updated later

st.markdown("---")

# -------------------------
# Fetch option chain via nsepython
# -------------------------
with st.spinner("Fetching option chain from NSE (may be rate-limited)..."):
    try:
        oc = nse_optionchain_scrapper(symbol_choice)
    except Exception as e:
        st.error(f"Option-chain fetch failed: {e}")
        oc = None

if oc is None or "records" not in oc:
    st.warning("Option-chain not available. NSE may block requests. Try later.")
    st.stop()

records = oc["records"].get("data", [])
expiry_list = oc["records"].get("expiryDates", [])
expiry_nearest = expiry_list[0] if expiry_list else None

# -------------------------
# Parse option chain -> rows list
# -------------------------
rows = []
for r in records:
    # if expiry filtering, ensure same expiry (makes sense if more expiries are in chain)
    if expiry_nearest and r.get("expiryDate") != expiry_nearest:
        continue
    strike = r.get("strikePrice")
    ce = r.get("CE") or {}
    pe = r.get("PE") or {}
    rows.append({
        "strike": strike,
        "CE_OI": int(ce.get("openInterest") or 0),
        "PE_OI": int(pe.get("openInterest") or 0),
        "CE_Chng_OI": int(ce.get("changeinOpenInterest") or 0),
        "PE_Chng_OI": int(pe.get("changeinOpenInterest") or 0),
        "CE_ltp": float(ce.get("lastPrice") or np.nan),
        "PE_ltp": float(pe.get("lastPrice") or np.nan),
    })

df_oc = pd.DataFrame(rows).drop_duplicates(subset="strike").sort_values("strike").reset_index(drop=True)
if df_oc.empty:
    st.warning("No option chain rows parsed.")
    st.stop()

# -------------------------
# Filter ATM ± buffer
# -------------------------
df_filtered = df_oc[(df_oc["strike"] >= atm_strike - buffer) & (df_oc["strike"] <= atm_strike + buffer)].copy()
if df_filtered.empty:
    st.warning("No strikes in selected ATM ± buffer range.")
    st.stop()

# compute ratios and PCR
df_filtered["PE/CE_Chng_OI_Ratio"] = (df_filtered["PE_Chng_OI"] / df_filtered["CE_Chng_OI"]).replace([np.inf, -np.inf], np.nan).round(2)
df_filtered["PCR_OI"] = (df_filtered["PE_OI"] / df_filtered["CE_OI"]).replace([np.inf, -np.inf], np.nan).round(2)

total_pe = int(df_filtered["PE_OI"].sum())
total_ce = int(df_filtered["CE_OI"].sum())
pcr_total = round(total_pe / max(total_ce, 1), 2)

# update PCR metric in header (replace col4)
col4.metric("PCR (OI, ATM±{})".format(buffer), f"{pcr_total:.2f}")

# -------------------------
# ATM highlight function
# -------------------------
def highlight_atm(row):
    return ['background-color: yellow' if row["strike"] == atm_strike else '' for _ in row]

# display option table
st.subheader(f"Option Chain — {symbol_choice} ATM ± {buffer}")
st.dataframe(
    df_filtered.rename(columns={
        "strike": "Strike",
        "CE_OI": "CE OI",
        "PE_OI": "PE OI",
        "CE_Chng_OI": "CE ΔOI",
        "PE_Chng_OI": "PE ΔOI",
        "CE_ltp": "CE LTP",
        "PE_ltp": "PE LTP",
        "PE/CE_Chng_OI_Ratio": "PE/CE ΔOI Ratio",
        "PCR_OI": "PCR (strike)"
    }).style.apply(highlight_atm, axis=1).format({
        "CE OI": "{:,}", "PE OI": "{:,}", "CE ΔOI": "{:+,}", "PE ΔOI": "{:+,}",
        "CE LTP": "{:.2f}", "PE LTP": "{:.2f}", "PE/CE ΔOI Ratio": "{:.2f}", "PCR (strike)": "{:.2f}"
    })
)

# -------------------------
# Top movers
# -------------------------
st.subheader("Top OI Movers (ATM range)")
df_filtered["Abs_CE_Chng"] = df_filtered["CE_Chng_OI"].abs()
df_filtered["Abs_PE_Chng"] = df_filtered["PE_Chng_OI"].abs()

top_ce = df_filtered.sort_values("Abs_CE_Chng", ascending=False).head(top_n)[["strike", "CE_Chng_OI", "CE_OI"]]
top_pe = df_filtered.sort_values("Abs_PE_Chng", ascending=False).head(top_n)[["strike", "PE_Chng_OI", "PE_OI"]]

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Top CE movers (by |ΔOI|)**")
    st.dataframe(top_ce.rename(columns={"strike":"Strike","CE_Chng_OI":"ΔOI","CE_OI":"OI"}).style.format({"ΔOI":"{:+,}","OI":"{:,}"}))
with c2:
    st.markdown("**Top PE movers (by |ΔOI|)**")
    st.dataframe(top_pe.rename(columns={"strike":"Strike","PE_Chng_OI":"ΔOI","PE_OI":"OI"}).style.format({"ΔOI":"{:+,}","OI":"{:,}"}))

# -------------------------
# IV (approx) calculation for filtered strikes (may be slow)
# -------------------------
st.subheader("Approximate IV (from LTP) — best-effort")
# compute time to expiry in years (approx)
def compute_T_from_expiry_list(expiry_list):
    # use first expiry if available
    if expiry_list:
        try:
            exp_str = expiry_list[0]
            exp_dt = pd.to_datetime(exp_str, format="%d-%b-%Y", errors='coerce')
            if pd.isna(exp_dt):
                # try other parse
                exp_dt = pd.to_datetime(exp_str, errors='coerce')
            # expiry at 15:30 IST -> convert to UTC approx
            now_utc = pd.Timestamp.now(tz="UTC")
            exp_local = pd.Timestamp(exp_dt.date()) + pd.Timedelta(hours=15, minutes=30)
            exp_utc = exp_local - pd.Timedelta(hours=5, minutes=30)
            T = max((exp_utc - now_utc).total_seconds() / (365*24*3600), 1e-6)
            return T
        except Exception:
            return 7/365.0
    return 7/365.0

T = compute_T_from_expiry_list(expiry_list)
ivs_ce = []
ivs_pe = []

for _, row in df_filtered.iterrows():
    K = row["strike"]
    # CE
    ce_iv = np.nan
    try:
        ce_price = row["CE_ltp"]
        if not (np.isnan(ce_price) or ce_price <= 0):
            ce_iv = implied_vol(ce_price, spot, K, T, r=0.0, option_type="C")
    except Exception:
        ce_iv = np.nan
    # PE
    pe_iv = np.nan
    try:
        pe_price = row["PE_ltp"]
        if not (np.isnan(pe_price) or pe_price <= 0):
            pe_iv = implied_vol(pe_price, spot, K, T, r=0.0, option_type="P")
    except Exception:
        pe_iv = np.nan
    ivs_ce.append(ce_iv if not np.isnan(ce_iv) else np.nan)
    ivs_pe.append(pe_iv if not np.isnan(pe_iv) else np.nan)

df_filtered["IV_CE"] = np.round(ivs_ce, 4)
df_filtered["IV_PE"] = np.round(ivs_pe, 4)
df_filtered["IV_Skew"] = (df_filtered["IV_PE"] - df_filtered["IV_CE"]).round(4)

# show IV table
st.dataframe(df_filtered[["strike","IV_CE","IV_PE","IV_Skew"]].rename(columns={"strike":"Strike"}).style.format({"IV_CE":"{:.2f}", "IV_PE":"{:.2f}", "IV_Skew":"{:.4f}"}))

# -------------------------
# Max Pain calculation
# -------------------------
st.subheader("Max Pain (nearest expiry) — estimated")
# compute total payout for each candidate settlement price equal to strike values
candidates = df_oc["strike"].unique()
payouts = []
for S in candidates:
    # call payout: sum max(0, S - K) * CE_OI
    call_payout = ((np.maximum(S - df_oc["strike"], 0)) * df_oc["CE_OI"]).sum()
    put_payout = ((np.maximum(df_oc["strike"] - S, 0)) * df_oc["PE_OI"]).sum()
    total_payout = call_payout + put_payout
    payouts.append((S, total_payout))
# pick strike with minimum payout
payouts_df = pd.DataFrame(payouts, columns=["Strike","TotalPayout"])
max_pain_strike = int(payouts_df.loc[payouts_df["TotalPayout"].idxmin(),"Strike"])
st.write(f"**Max Pain (estimated): {max_pain_strike}**")

# -------------------------
# PCR intraday trend (session)
# -------------------------
st.subheader("PCR (ATM range) — Intraday trend (session)")

if "pcr_history" not in st.session_state:
    st.session_state.pcr_history = []

if auto_append:
    now_hm = datetime.now().strftime("%H:%M:%S")
    # append only if last timestamp different to avoid duplicates on re-renders
    if len(st.session_state.pcr_history) == 0 or st.session_state.pcr_history[-1]["time"] != now_hm:
        st.session_state.pcr_history.append({"time": now_hm, "pcr": pcr_total})

# build dataframe and display chart
pcr_df = pd.DataFrame(st.session_state.pcr_history)
if not pcr_df.empty:
    pcr_df = pcr_df.drop_duplicates(subset="time").reset_index(drop=True)
    # show line chart with neutral line at 1
    fig_pcr = go.Figure()
    fig_pcr.add_trace(go.Scatter(x=pcr_df["time"], y=pcr_df["pcr"], mode="lines+markers", name="PCR (ATM range)"))
    fig_pcr.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="Neutral (1)", annotation_position="bottom right")
    fig_pcr.update_layout(yaxis_title="PCR", xaxis_title="Time", template="plotly_dark", height=350)
    st.plotly_chart(fig_pcr, use_container_width=True)

    # download button
    csv_bytes = pcr_df.to_csv(index=False).encode()
    st.download_button("Download PCR trend CSV", csv_bytes, file_name=f"pcr_trend_{symbol_choice}_{datetime.now().date()}.csv", mime="text/csv")
else:
    st.info("PCR history is empty for this session. Enable 'Append PCR to intraday trend' to collect data automatically.")

# -------------------------
# PCR gauge for quick view
# -------------------------
st.subheader("PCR Gauge (ATM range)")
gcolor = "green" if pcr_total > 1 else "red" if pcr_total < 1 else "gray"
fig_g = go.Figure(go.Indicator(mode="gauge+number", value=float(pcr_total), title={"text":"PCR (OI)"}, gauge={
    "axis":{"range":[0, max(2.0, pcr_total+0.5)]},
    "steps":[{"range":[0,1],"color":"red"},{"range":[1,2],"color":"green"}],
    "bar":{"color":gcolor}
}))
st.plotly_chart(fig_g, use_container_width=True)

st.caption("Notes: IVs are approximations derived from LTPs and Black–Scholes; results depend on LTP availability. NSE option-chain access may be rate-limited. Use for research/educational purposes only.")
