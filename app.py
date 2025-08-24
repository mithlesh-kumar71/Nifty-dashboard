# app.py
# ---------------------------------------------
# NSE Options & Charting Dashboard (Streamlit)
# Features:
# - Auto-refresh (1s to 120s)
# - Chart: Line or Candlestick toggle
# - Timeframes: 1m, 5m, 15m, 60m, 1d
# - Historical data: period OR custom date range
# - Supertrend overlay (native implementation)
# - India VIX widget
# - Optional NSE option-chain snapshot: PCR, ATM Straddle, ATM IV (best-effort)
# ---------------------------------------------

import math
from datetime import datetime, timedelta, date, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# ------------------------------
# UI: Page setup
# ------------------------------
st.set_page_config(page_title="NSE Options & Charts", layout="wide")
st.title("📊 NSE Options & Charts — Live")

with st.sidebar:
    st.header("Controls")

    # Symbol & Board
    board = st.selectbox("Board / Index", ["NIFTY", "BANKNIFTY"], index=0)
    default_symbol = "^NSEI" if board == "NIFTY" else "^NSEBANK"
    symbol = st.text_input("Chart symbol (Yahoo Finance)", default_symbol, help="Examples: ^NSEI (NIFTY), ^NSEBANK (BANKNIFTY), RELIANCE.NS, TCS.NS")

    # Timeframe & chart type
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "60m", "1d"], index=0)
    chart_type = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)

    # Historical data mode
    hist_mode = st.radio("Historical mode", ["Period", "Custom Dates"], index=0)

    if timeframe == "1m":
        period_default = "5d"
        period_choices = ["1d", "5d", "7d"]
    elif timeframe in ["5m", "15m", "60m"]:
        period_default = "1mo"
        period_choices = ["5d", "1mo", "3mo", "6mo", "1y"]
    else:  # 1d
        period_default = "6mo"
        period_choices = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]

    if hist_mode == "Period":
        period = st.selectbox("Period", period_choices, index=period_choices.index(period_default))
        start = end = None
    else:
        today = date.today()
        start = st.date_input("Start date", today - timedelta(days=60))
        end = st.date_input("End date", today)
        period = None

    # Supertrend params
    st.subheader("Supertrend")
    st_help = "Typical values: period=10, multiplier=3. Larger period/smaller multiplier = more signals."
    st_period = st.number_input("ATR Period", min_value=7, max_value=50, value=10, step=1, help=st_help)
    st_mult = st.number_input("Multiplier", min_value=1.0, max_value=6.0, value=3.0, step=0.5, help=st_help)
    show_supertrend = st.toggle("Show Supertrend", value=True)

    # Refresh
    st.subheader("Auto-Refresh")
    refresh_sec = st.slider("Interval (seconds)", min_value=1, max_value=120, value=5, step=1)
    st.caption("⚠️ Very fast refresh may cause throttling by data sources (try ≥ 5–10s).")

# ------------------------------
# Auto-refresh (JS reload)
# ------------------------------
st.markdown(
    f"""
    <script>
    setTimeout(function() {{
        window.location.reload();
    }}, {int(refresh_sec) * 1000});
    </script>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Helpers: Technical indicators
# ------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, period: int = 10, ema: bool = True) -> pd.Series:
    tr = true_range(df)
    if ema:
        return tr.ewm(span=period, adjust=False).mean()
    return tr.rolling(period).mean()

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Returns df with columns:
    - 'ST'  : supertrend line
    - 'Trend' : +1 uptrend, -1 downtrend
    """
    out = df.copy()
    _atr = atr(out, period=period, ema=True)
    hl2 = (out["High"] + out["Low"]) / 2.0
    upperband = hl2 + multiplier * _atr
    lowerband = hl2 - multiplier * _atr

    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()

    for i in range(1, len(out)):
        final_upperband.iat[i] = (
            upperband.iat[i]
            if (upperband.iat[i] < final_upperband.iat[i - 1]) or (out["Close"].iat[i - 1] > final_upperband.iat[i - 1])
            else final_upperband.iat[i - 1]
        )
        final_lowerband.iat[i] = (
            lowerband.iat[i]
            if (lowerband.iat[i] > final_lowerband.iat[i - 1]) or (out["Close"].iat[i - 1] < final_lowerband.iat[i - 1])
            else final_lowerband.iat[i - 1]
        )

    trend = np.ones(len(out), dtype=int)
    st_line = pd.Series(index=out.index, dtype=float)

    for i in range(len(out)):
        if i == 0:
            trend[i] = 1
            st_line.iat[i] = final_lowerband.iat[i]
        else:
            if (out["Close"].iat[i] > final_upperband.iat[i - 1]):
                trend[i] = 1
            elif (out["Close"].iat[i] < final_lowerband.iat[i - 1]):
                trend[i] = -1
            else:
                trend[i] = trend[i - 1]

            if trend[i] == 1:
                st_line.iat[i] = final_lowerband.iat[i]
            else:
                st_line.iat[i] = final_upperband.iat[i]

    out["ST"] = st_line
    out["Trend"] = trend
    return out

# ------------------------------
# Data fetchers
# ------------------------------

def fetch_yf(symbol: str, interval: str, period: str = None, start: date = None, end: date = None) -> pd.DataFrame:
    """Fetch OHLC data from yfinance using either period or start/end."""
    kwargs = dict(interval=interval, auto_adjust=False, prepost=False, progress=False)
    if period is not None:
        df = yf.download(symbol, period=period, **kwargs)
    else:
        # yfinance expects datetime-like
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end) + pd.Timedelta(days=1)  # inclusive
        df = yf.download(symbol, start=start_dt, end=end_dt, **kwargs)
    df = df.dropna()
    return df

def fetch_vix_last() -> float | None:
    try:
        vix = yf.Ticker("^INDIAVIX").history(period="5d", interval="5m")["Close"]
        if len(vix):
            return float(vix.iloc[-1])
    except Exception:
        pass
    return None

# NSE option chain (best-effort; may be throttled)
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
    try:
        s.get("https://www.nseindia.com", timeout=10)
    except Exception:
        pass
    return s

def fetch_option_chain(symbol="NIFTY"):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    s = nse_session()
    r = s.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

def parse_option_chain(json_data):
    rec = json_data.get("records", {})
    underlying = rec.get("underlyingValue")
    expiries = rec.get("expiryDates", [])
    data = rec.get("data", [])
    rows = []
    for row in data:
        strike = row.get("strikePrice")
        expiry = row.get("expiryDate")
        ce = row.get("CE", {}) or {}
        pe = row.get("PE", {}) or {}
        rows.append({
            "expiry": expiry, "strike": strike,
            "CE_bid": ce.get("bidprice"), "CE_ask": ce.get("askPrice"),
            "CE_ltp": ce.get("lastPrice"), "CE_oi": ce.get("openInterest"),
            "PE_bid": pe.get("bidprice"), "PE_ask": pe.get("askPrice"),
            "PE_ltp": pe.get("lastPrice"), "PE_oi": pe.get("openInterest"),
        })
    return underlying, expiries, pd.DataFrame(rows)

def nearest_atm_strike(spot: float, strikes: list[int|float]) -> float:
    return min(strikes, key=lambda k: abs(k - spot))

def bs_price(S, K, T, r, q, sigma, option_type="C"):
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if option_type == "C" else (K - S))
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    from math import erf, sqrt
    N = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    if option_type == "C":
        return S * math.exp(-q*T) * N(d1) - K * math.exp(-0*T) * N(d2)
    else:
        return K * math.exp(-0*T) * N(-d2) - S * math.exp(-q*T) * N(-d1)

def implied_vol(target_price, S, K, T, r=0.0, q=0.0, typ="C", tol=1e-6, iters=100):
    if not (target_price and S and K and T) or target_price <= 0:
        return np.nan
    lo, hi = 1e-6, 5.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        px = bs_price(S, K, T, r, q, mid, typ)
        if abs(px - target_price) < tol:
            return mid
        if px > target_price:
            hi = mid
        else:
            lo = mid
    return np.nan

# ------------------------------
# Fetch chart data (main panel)
# ------------------------------
try:
    if hist_mode == "Period":
        df = fetch_yf(symbol, timeframe, period=period)
    else:
        df = fetch_yf(symbol, timeframe, start=start, end=end)
except Exception as e:
    st.error(f"Failed to fetch price data: {e}")
    df = pd.DataFrame()

# ------------------------------
# KPI row
# ------------------------------
k1, k2, k3, k4 = st.columns(4)

last_close = float(df["Close"].iloc[-1]) if len(df) else None
vix_last = fetch_vix_last()

k1.metric("Symbol", symbol)
k2.metric("Last Price", f"{last_close:,.2f}" if last_close else "—")
k3.metric("India VIX", f"{vix_last:,.2f}" if vix_last else "—")
k4.metric("Timeframe", timeframe)

st.divider()

# ------------------------------
# Build chart with toggle & supertrend
# ------------------------------
st.subheader("Price Chart")

if df.empty:
    st.info("No price data available for the chosen inputs.")
else:
    # Compute supertrend if requested
    if show_supertrend:
        df_st = supertrend(df[["Open", "High", "Low", "Close"]].copy(), period=int(st_period), multiplier=float(st_mult))
    else:
        df_st = None

    fig = go.Figure()

    if chart_type == "Line":
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["Close"], mode="lines", name="Close"
            )
        )
    else:
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Candles",
            )
        )
        fig.update_xaxes(rangeslider_visible=False)

    # Overlay supertrend
    if show_supertrend and df_st is not None and len(df_st):
        # Split up/down segments so the color reflects trend
        st_up = df_st["ST"].where(df_st["Trend"] == 1)
        st_dn = df_st["ST"].where(df_st["Trend"] == -1)

        fig.add_trace(go.Scatter(x=df.index, y=st_up, mode="lines", name="Supertrend (Up)"))
        fig.add_trace(go.Scatter(x=df.index, y=st_dn, mode="lines", name="Supertrend (Down)"))

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=500,
        xaxis_title="Time",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Option-chain snapshot (best-effort)
# ------------------------------
st.divider()
st.subheader(f"{board} Option-Chain Snapshot (PCR • ATM Straddle • ATM IV)")

pcr_val = np.nan
atm_strike = None
atm_straddle = np.nan
atm_ce_iv = atm_pe_iv = np.nan
ce_ba = pe_ba = (np.nan, np.nan)

try:
    oc_json = fetch_option_chain(board)
    spot, expiries, oc_df = parse_option_chain(oc_json)
    expiry = expiries[0] if expiries else None

    if not oc_df.empty and expiry and spot:
        dfe = oc_df[oc_df["expiry"] == expiry].copy()
        pcr_val = dfe["PE_oi"].fillna(0).sum() / max(dfe["CE_oi"].fillna(0).sum(), 1)

        strikes = sorted(dfe["strike"].dropna().unique())
        if strikes:
            atm_strike = nearest_atm_strike(spot, strikes)
            row = dfe[dfe["strike"] == atm_strike].iloc[0]
            ce_ltp, pe_ltp = row["CE_ltp"], row["PE_ltp"]
            ce_ba = (row["CE_bid"], row["CE_ask"])
            pe_ba = (row["PE_bid"], row["PE_ask"])
            atm_straddle = (ce_ltp or 0) + (pe_ltp or 0)

            # Expiry assumed 15:30 IST -> UTC
            try:
                exp_dt = datetime.strptime(expiry, "%d-%b-%Y").replace(hour=15, minute=30)
                exp_dt_utc = exp_dt - timedelta(hours=5, minutes=30)
            except Exception:
                exp_dt_utc = datetime.utcnow()

            T = max((exp_dt_utc - datetime.now(timezone.utc)).total_seconds() / (365*24*3600), 1e-6)
            atm_ce_iv = implied_vol(ce_ltp, spot, atm_strike, T, 0.0, 0.0, "C")
            atm_pe_iv = implied_vol(pe_ltp, spot, atm_strike, T, 0.0, 0.0, "P")
except Exception as e:
    st.caption("⚠️ Option-chain fetch may be throttled by NSE; try a longer refresh interval if this section is empty.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot", f"{spot:,.2f}" if 'spot' in locals() and spot else "—")
c2.metric("PCR (OI)", f"{pcr_val:,.2f}" if not np.isnan(pcr_val) else "—")
c3.metric("ATM Straddle", f"{atm_straddle:,.2f}" if not np.isnan(atm_straddle) else "—")
c4.metric("ATM IV (CE / PE)", f"{(atm_ce_iv*100):.2f}% / {(atm_pe_iv*100):.2f}%" if not (np.isnan(atm_ce_iv) or np.isnan(atm_pe_iv)) else "—")

# Bid-Ask details
if atm_strike:
    st.write(
        f"**Nearest ATM:** {int(atm_strike)} | **CE Bid/Ask:** {ce_ba} | **PE Bid/Ask:** {pe_ba}"
    )

st.caption("Data sources: Yahoo Finance (prices/VIX), NSE (option chain). Educational use only — may be delayed or rate-limited.")
