# app.py
# Robust Streamlit dashboard: Supertrend + simple signals for 6 symbols
# Minimal deps: streamlit, pandas, numpy, yfinance, plotly

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# ---------- Page setup (must be the first Streamlit command) ----------
st.set_page_config(page_title="Simple Supertrend Signals", layout="wide")
st.title("📈 Supertrend Signals — NIFTY + Selected Stocks (stable build)")

# ---------- Watchlist (the 6 you asked for) ----------
SYMBOLS = {
    "NIFTY": "^NSEI",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "SBI": "SBIN.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "TATAMOTORS": "TATAMOTORS.NS"
}

# ---------- UI controls ----------
col1, col2, col3 = st.columns([3,2,2])
with col1:
    sym_name = st.selectbox("Symbol", list(SYMBOLS.keys()), index=0)
with col2:
    # Daily is default because Yahoo intraday for Indian stocks is flaky
    interval = st.selectbox("Interval", ["1d", "60m", "30m"], index=0)
with col3:
    period = st.selectbox("History period (yfinance)", ["6mo", "1y", "2y"], index=0)

chart_type = st.radio("Chart type", ["Candlestick", "Line"], horizontal=True)

# ---------- Helpers: fetch with MultiIndex-safety and validation ----------
@st.cache_data(ttl=60)
def fetch_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV and normalize columns. Return empty df on failure."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # If columns are MultiIndex (happens for some intervals), flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, c)).strip() for c in df.columns.values]

    # Standard expected names: Open, High, Low, Close, Volume
    # Try to coerce those columns if they exist with suffixes
    col_map = {}
    for base in ["Open","High","Low","Close","Volume"]:
        if base in df.columns:
            col_map[base] = base
        else:
            # try case-insensitive match
            matches = [c for c in df.columns if c.lower().startswith(base.lower())]
            if matches:
                col_map[matches[0]] = base

    # If no Close-like column found, fail
    if "Close" not in col_map.values():
        return pd.DataFrame()

    # rename matched columns to canonical names
    df = df.rename(columns=col_map)

    # keep canonical columns and drop rows with NaN close
    expected = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[expected].dropna(subset=["Close"])

    # Ensure numeric types
    df[expected] = df[expected].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Close"])
    if df.empty:
        return pd.DataFrame()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df = df.set_index("Date")
        else:
            df.index = pd.to_datetime(df.index)
    df.index = pd.to_datetime(df.index)

    return df

# ---------- Indicator implementations (robust, vectorized) ----------
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-smoothed ATR via ewm on true range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """
    Return Supertrend line (st) and trend series (+1 up, -1 down).
    Implemented with numpy arrays, returns Series aligned to df.index.
    """
    if df.empty or len(df) < period + 2:
        return pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=int)

    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)

    atr = compute_atr(df, period=period).to_numpy(dtype=float)
    hl2 = (high + low) / 2.0
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    final_upper = upper.copy()
    final_lower = lower.copy()
    n = len(df)

    for i in range(1, n):
        if (upper[i] < final_upper[i-1]) or (close[i-1] > final_upper[i-1]):
            final_upper[i] = upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        if (lower[i] > final_lower[i-1]) or (close[i-1] < final_lower[i-1]):
            final_lower[i] = lower[i]
        else:
            final_lower[i] = final_lower[i-1]

    trend = np.ones(n, dtype=int)
    st_line = np.zeros(n, dtype=float)

    for i in range(n):
        if i == 0:
            trend[i] = 1
            st_line[i] = final_lower[i]
            continue
        if close[i] > final_upper[i-1]:
            trend[i] = 1
        elif close[i] < final_lower[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

        st_line[i] = final_lower[i] if trend[i] == 1 else final_upper[i]

    st_series = pd.Series(st_line, index=df.index)
    trend_series = pd.Series(trend, index=df.index)
    return st_series, trend_series

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    cum_pv = pv.cumsum()
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    return cum_pv / cum_vol

# ---------- Safe pipeline: fetch -> add indicators ----------
ticker = SYMBOLS[sym_name]
raw = fetch_ohlcv(ticker, period=period, interval=interval)

if raw.empty:
    st.error(f"No data available for {sym_name} with interval={interval} and period={period}. Try interval=1d or change period.")
    st.stop()

df = raw.copy()
# add indicators
df["ATR"] = compute_atr(df, period=14)
df["VWAP"] = compute_vwap(df)
df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
df["RSI"] = (df["Close"].diff().clip(lower=0).ewm(alpha=1/14, adjust=False).mean() /
             df["Close"].diff().abs().ewm(alpha=1/14, adjust=False).mean() * 100)
st_series, trend_series = compute_supertrend(df, period=10, multiplier=3.0)
df["ST"] = st_series
df["Trend"] = trend_series

# ---------- Simple rule-based signal ----------
def row_signal(r):
    try:
        if np.isnan(r["ST"]):
            return "NO DATA"
        # simple confluence
        if (r["Trend"] == 1) and (r["Close"] > r["VWAP"]) and (r["EMA9"] > r["EMA21"]) and (r["RSI"] > 50):
            return "LONG"
        if (r["Trend"] == -1) and (r["Close"] < r["VWAP"]) and (r["EMA9"] < r["EMA21"]) and (r["RSI"] < 50):
            return "SHORT"
        return "NO TRADE"
    except Exception:
        return "NO DATA"

latest = df.iloc[-1]
signal = row_signal(latest)

# ---------- Top KPIs ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Symbol", sym_name)
k2.metric("Last Price", f"{latest['Close']:.2f}")
k3.metric("Signal", signal)
k4.metric("RSI", f"{latest['RSI']:.2f}" if not np.isnan(latest["RSI"]) else "—")

st.markdown("**Rule:** Supertrend trend + VWAP + EMA9/EMA21 + RSI filter")

# ---------- Plotly chart ----------
st.subheader("Price chart")
fig = go.Figure()
if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ))
else:
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))

fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], mode="lines", name="VWAP"))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], mode="lines", name="EMA9"))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], mode="lines", name="EMA21"))
fig.add_trace(go.Scatter(x=df.index, y=df["ST"], mode="lines", name="Supertrend"))

# show markers where signal is LONG/SHORT
sig_mask = df.apply(lambda r: row_signal(r) in ("LONG","SHORT"), axis=1)
sig_pts = df[sig_mask]
if not sig_pts.empty:
    colors = ["green" if row_signal(r) == "LONG" else "red" for _, r in sig_pts.iterrows()]
    fig.add_trace(go.Scatter(x=sig_pts.index, y=sig_pts["Close"], mode="markers",
                             marker=dict(color=colors, size=8), name="Signals"))

fig.update_layout(template="plotly_dark", height=650, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ---------- Recent table ----------
st.subheader("Recent data (last 20 rows)")
display_cols = ["Close","VWAP","EMA9","EMA21","RSI","ATR","ST","Trend"]
st.dataframe(df[display_cols].tail(20).assign(Time=lambda d: d.index.strftime("%Y-%m-%d")))

# ---------- Quick watchlist scan (6 symbols) ----------
st.subheader("Quick watchlist scan (6 symbols)")
scan = []
for name, tk in SYMBOLS.items():
    d = fetch_ohlcv(tk, period="3mo", interval="1d")
    if d.empty:
        scan.append({"Symbol": name, "Price": None, "Signal": "NO DATA"})
        continue
    # compute light indicators
    d["VWAP"] = compute_vwap(d)
    d["EMA9"] = d["Close"].ewm(span=9, adjust=False).mean()
    d["EMA21"] = d["Close"].ewm(span=21, adjust=False).mean()
    st_s, tr_s = compute_supertrend(d, period=10, multiplier=3.0)
    d["ST"] = st_s; d["Trend"] = tr_s
    last = d.iloc[-1]
    scan.append({"Symbol": name, "Price": float(last["Close"]), "Signal": row_signal(last)})
scan_df = pd.DataFrame(scan)
st.dataframe(scan_df)

st.caption("This app uses Yahoo Finance (yfinance). Intraday for NSE can be unreliable on Yahoo — use daily data for best stability.")
