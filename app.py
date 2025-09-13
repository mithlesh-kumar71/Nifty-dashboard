# app.py
# Intraday Options Dashboard — Enhanced
# - Price chart with VWAP + Supertrend + RSI
# - ATM ± buffer option-chain, PCR, IV skew (approx)
# - OI change heatmap (CE/PE) and top movers
# - PCR trend logging (Excel) + chart
# - India VIX (Yahoo) display
# - Auto-refresh control
# NOTE: Works best when required packages are installed (see comments above)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import math
import os

# NSE helper
from nsepython import nse_optionchain_scrapper  # nsepython package
# Excel writer
import io

# -------------------------
# Page config (must be first Streamlit call)
# -------------------------
st.set_page_config(page_title="Intraday Options + OI Heatmap", layout="wide")

# -------------------------
# Top controls
# -------------------------
st.title("📈 Intraday Options Dashboard — OI Heatmap • PCR • IV Skew")

col1, col2 = st.columns([1, 1])
with col1:
    symbol_choice = st.selectbox("Index", ["NIFTY", "BANKNIFTY"], index=0)
with col2:
    refresh_sec = st.slider("Auto-refresh (sec)", min_value=30, max_value=900, value=300, step=30)

# manual button to force refresh
if st.button("Refresh Now"):
    st.experimental_rerun()

# -------------------------
# Helpers: indicators & BS-IV solver
# -------------------------
def atr(series_high, series_low, series_close, period=14):
    high_low = series_high - series_low
    high_close = (series_high - series_close.shift()).abs()
    low_close = (series_low - series_close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def supertrend(df, period=10, multiplier=3.0):
    out = df.copy()
    _atr = atr(out["High"], out["Low"], out["Close"], period)
    hl2 = (out["High"] + out["Low"]) / 2
    upperband = hl2 + multiplier * _atr
    lowerband = hl2 - multiplier * _atr
    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    for i in range(1, len(out)):
        final_upper.iat[i] = (upperband.iat[i]
                              if (upperband.iat[i] < final_upper.iat[i-1]) or (out["Close"].iat[i-1] > final_upper.iat[i-1])
                              else final_upper.iat[i-1])
        final_lower.iat[i] = (lowerband.iat[i]
                              if (lowerband.iat[i] > final_lower.iat[i-1]) or (out["Close"].iat[i-1] < final_lower.iat[i-1])
                              else final_lower.iat[i-1])
    trend = np.ones(len(out), dtype=int)
    stline = pd.Series(index=out.index, dtype=float)
    for i in range(len(out)):
        if i == 0:
            trend[i] = 1
            stline.iat[i] = final_lower.iat[i]
        else:
            if out["Close"].iat[i] > final_upper.iat[i-1]:
                trend[i] = 1
            elif out["Close"].iat[i] < final_lower.iat[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
            stline.iat[i] = final_lower.iat[i] if trend[i] == 1 else final_upper.iat[i]
    out["ST"] = stline
    out["Trend"] = trend
    return out

# Black-Scholes (European) call/put price (no dividends, r ~ 0)
def bs_price(S, K, T, r, sigma, option_type="C"):
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if option_type == "C" else (K - S))
    from math import log, sqrt, exp
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    # normal cdf
    from math import erf, sqrt as _sqrt
    N = lambda x: 0.5 * (1.0 + erf(x / _sqrt(2.0)))
    if option_type == "C":
        return S * N(d1) - K * math.exp(-r * T) * N(d2)
    else:
        return K * math.exp(-r * T) * N(-d2) - S * N(-d1)

def implied_vol(target_price, S, K, T, r=0.0, option_type="C"):
    # simple bisection
    if target_price is None or target_price <= 0:
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

# utility: rounded ATM strike to nearest 50 or 100 depending on index
def round_strike(spot, idx):
    step = 50 if idx == "NIFTY" else 100
    return int(round(spot / step) * step)

# -------------------------
# Fetch price & indicators
# -------------------------
with st.spinner("Fetching price data..."):
    yf_symbol = "^NSEI" if symbol_choice == "NIFTY" else "^NSEBANK"
    try:
        # use 5m interval price to be general
        price_df = yf.Ticker(yf_symbol).history(period="5d", interval="5m").reset_index().dropna()
    except Exception as e:
        st.error(f"Price fetch failed: {e}")
        price_df = pd.DataFrame()

if price_df.empty:
    st.error("No price data. Try again or adjust period/interval.")
    st.stop()

# compute indicators
price_df = price_df.rename(columns={"Datetime": "Date"}) if "Datetime" in price_df.columns else price_df
price_df = price_df.sort_values("Date").reset_index(drop=True)
price_df = supertrend(price_df)
# simple RSI via reasoned approach (14)
delta = price_df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss
price_df["RSI"] = 100 - (100 / (1 + rs))
price_df["VWAP"] = (price_df["Close"] * price_df["Volume"]).cumsum() / price_df["Volume"].cumsum()

# -------------------------
# Top row: VIX, spot, ATM strike, PCR (later)
# -------------------------
col_a, col_b, col_c, col_d = st.columns(4)
spot = price_df["Close"].iloc[-1]
atm = round_strike(spot, symbol_choice)
# India VIX from Yahoo: ^INDIAVIX
try:
    vix = yf.Ticker("^INDIAVIX").history(period="5d", interval="1d")["Close"].iloc[-1]
    col_a.metric("India VIX", f"{vix:.2f}")
except Exception:
    col_a.metric("India VIX", "—")
col_b.metric("Spot", f"{spot:.2f}")
col_c.metric("ATM Strike", f"{atm}")
# placeholder for PCR — we'll fill after option-chain fetch
col_d.metric("PCR (ATM±225)", "—")

# -------------------------
# Price chart with VWAP + ST
# -------------------------
st.subheader("Price Chart (VWAP • Supertrend)")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=price_df["Date"], open=price_df["Open"], high=price_df["High"],
                             low=price_df["Low"], close=price_df["Close"], name="Price"))
fig.add_trace(go.Scatter(x=price_df["Date"], y=price_df["VWAP"], mode="lines", name="VWAP", line=dict(color="cyan")))
fig.add_trace(go.Scatter(x=price_df["Date"], y=price_df["ST"], mode="lines", name="Supertrend", line=dict(color="orange")))
fig.update_layout(template="plotly_dark", height=500, margin=dict(t=30))
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Option chain / OI heatmap / PCR / IV skew
# -------------------------
st.subheader("Option Chain (ATM ± buffer) • OI Heatmap & Top Movers")
buffer = st.sidebar.number_input("ATM buffer (points)", min_value=100, max_value=1000, step=50, value=300)
topn = st.sidebar.number_input("Top movers (N)", min_value=3, max_value=20, value=8)

with st.spinner("Fetching option chain (NSE)..."):
    try:
        oc = nse_optionchain_scrapper(symbol_choice)  # returns dict similar to before
    except Exception as e:
        st.error(f"NSE option-chain fetch failed: {e}")
        oc = None

if oc is None:
    st.warning("Option chain not available (NSE blocked or nsepython error). Showing only price data.")
else:
    records = oc.get("records", {}).get("data", [])
    # extract nearest expiry only
    expiry_dates = oc.get("records", {}).get("expiryDates", [])
    expiry = expiry_dates[0] if expiry_dates else None

    rows = []
    for d in records:
        try:
            if expiry and d.get("expiryDate") != expiry:
                continue
            strike = d.get("strikePrice")
            ce = d.get("CE") or {}
            pe = d.get("PE") or {}
            rows.append({
                "strike": strike,
                "CE_OI": ce.get("openInterest", 0),
                "PE_OI": pe.get("openInterest", 0),
                "CE_Chng_OI": ce.get("changeinOpenInterest", 0),
                "PE_Chng_OI": pe.get("changeinOpenInterest", 0),
                "CE_ltp": ce.get("lastPrice", np.nan),
                "PE_ltp": pe.get("lastPrice", np.nan),
            })
        except Exception:
            continue

    df_oc = pd.DataFrame(rows).drop_duplicates(subset="strike").sort_values("strike").reset_index(drop=True)
    if df_oc.empty:
        st.warning("Parsed option chain is empty.")
    else:
        # filter ATM ± buffer
        df_filtered = df_oc[(df_oc["strike"] >= atm - buffer) & (df_oc["strike"] <= atm + buffer)].copy()
        # compute ratios & clean
        df_filtered["PE/CE_Chng_OI_Ratio"] = (df_filtered["PE_Chng_OI"] / df_filtered["CE_Chng_OI"]).replace([np.inf, -np.inf], np.nan).round(2)
        df_filtered["PCR_OI"] = (df_filtered["PE_OI"] / df_filtered["CE_OI"]).replace([np.inf, -np.inf], np.nan).round(2)

        # PCR over filtered strikes
        total_pe = int(df_filtered["PE_OI"].sum())
        total_ce = int(df_filtered["CE_OI"].sum())
        pcr_total = round(total_pe / max(total_ce, 1), 2)

        # update PCR metric
        col_d.metric("PCR (OI, ATM±{})".format(buffer), f"{pcr_total:.2f}")

        # Top movers by absolute change in OI
        df_filtered["Abs_CE_Chng"] = df_filtered["CE_Chng_OI"].abs()
        df_filtered["Abs_PE_Chng"] = df_filtered["PE_Chng_OI"].abs()
        top_ce = df_filtered.sort_values("Abs_CE_Chng", ascending=False).head(topn)[["strike", "CE_Chng_OI", "CE_OI"]]
        top_pe = df_filtered.sort_values("Abs_PE_Chng", ascending=False).head(topn)[["strike", "PE_Chng_OI", "PE_OI"]]

        # OI Heatmap: create pivot of change OI per strike for CE and PE
        heat_df = df_filtered.set_index("strike")[["CE_Chng_OI", "PE_Chng_OI"]].T
        # heat_df rows: CE_Chng_OI and PE_Chng_OI; columns: strikes

        # plot heatmap
        st.markdown("**OI Change Heatmap (CE / PE)**")
        try:
            fig_heat = go.Figure(data=go.Heatmap(
                z=heat_df.values,
                x=heat_df.columns.astype(str),
                y=heat_df.index,
                colorscale="RdYlGn",  # red = negative, green = positive
                zmid=0
            ))
            fig_heat.update_layout(height=300, xaxis_title="Strike", yaxis_title="Side (CE/PE)")
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception as e:
            st.warning(f"Heatmap plot failed: {e}")

        # Top movers display
        st.markdown("**Top CE movers (by |ΔOI|)**")
        st.dataframe(top_ce.rename(columns={"strike":"Strike","CE_Chng_OI":"ΔOI","CE_OI":"OI"}).reset_index(drop=True).style.format({"ΔOI":"{:+,}","OI":"{:,}"}))
        st.markdown("**Top PE movers (by |ΔOI|)**")
        st.dataframe(top_pe.rename(columns={"strike":"Strike","PE_Chng_OI":"ΔOI","PE_OI":"OI"}).reset_index(drop=True).style.format({"ΔOI":"{:+,}","OI":"{:,}"}))

        # IV skew approx: compute IV per side if LTP available (approx)
        st.markdown("**IV (approx) around ATM**")
        # For IV we need time to expiry T (in years). approximate expiry at 15:30 IST on expiry date
        def compute_iv_row(row):
            S = spot
            K = row["strike"]
            # T: days to expiry
            T = 1/365.0
            # try derive days using expiry if available in oc dict (not always)
            # approximate T = 7 days if not present (best-effort)
            # We attempt to find expiry from oc
            try:
                expiry_list = oc.get("records", {}).get("expiryDates", [])
                if expiry_list:
                    # parse first expiry into datetime (format '30-Jul-2025') or similar
                    try:
                        exp_dt = pd.to_datetime(expiry_list[0], format="%d-%b-%Y")
                        # set time to 15:30 IST
                        now_utc = pd.Timestamp.now(tz="UTC")
                        exp_utc = pd.Timestamp(exp_dt.date()) + pd.Timedelta(hours=15, minutes=30) - pd.Timedelta(hours=5, minutes=30)
                        T = max((exp_utc - now_utc).total_seconds() / (365*24*3600), 1e-6)
                    except Exception:
                        T = 7/365.0
            except Exception:
                T = 7/365.0
            # try both CE and PE LTP
            iv_ce = implied = np.nan
            try:
                ce_price = row.get("CE_ltp", np.nan)
                if not (pd.isna(ce_price) or ce_price <= 0):
                    iv_ce = implied_vol(ce_price, S, K, T, r=0.0, option_type="C")
            except Exception:
                iv_ce = np.nan
            iv_pe = np.nan
            try:
                pe_price = row.get("PE_ltp", np.nan)
                if not (pd.isna(pe_price) or pe_price <= 0):
                    iv_pe = implied_vol(pe_price, S, K, T, r=0.0, option_type="P")
            except Exception:
                iv_pe = np.nan
            return iv_ce, iv_pe

        # compute ivs for filtered strikes (may be slow)
        ivs = [compute_iv_row(row) for _, row in df_filtered.iterrows()]
        if ivs:
            df_filtered["IV_CE"], df_filtered["IV_PE"] = zip(*ivs)
            df_filtered["IV_Skew"] = (df_filtered["IV_PE"] - df_filtered["IV_CE"]).round(4)
            st.dataframe(df_filtered[["strike","IV_CE","IV_PE","IV_Skew"]].rename(columns={"strike":"Strike"}).style.format({"IV_CE":"{:.2f}","IV_PE":"{:.2f}","IV_Skew":"{:.4f}"}))
        else:
            st.info("IV not computed (no LTPs available).")

        # -------------------------
        # PCR intraday trend saving & plotting
        # -------------------------
        st.markdown("**PCR Intraday Trend (ATM)**")
        # maintain PCR history in session state (list of dicts)
        if "pcr_history" not in st.session_state:
            st.session_state["pcr_history"] = []
        current_time = datetime.now().strftime("%H:%M")
        st.session_state["pcr_history"].append({"time": current_time, "pcr": pcr_total})
        pcr_df = pd.DataFrame(st.session_state["pcr_history"])
        # keep only today's entries
        st.line_chart(pcr_df.set_index("time")["pcr"])

        # download trend as excel/csv
        csv_bytes = pcr_df.to_csv(index=False).encode()
        st.download_button("Download PCR trend CSV", csv_bytes, file_name=f"pcr_trend_{datetime.now().date()}.csv", mime="text/csv")

# -------------------------
# Footer / help
# -------------------------
st.caption("Notes: NSE option-chain data may be rate-limited. IV is approximate and depends on availability of option LTP. Use for educational purposes only.")
