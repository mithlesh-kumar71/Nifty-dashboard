# app.py
# Robust Streamlit dashboard: Supertrend + simple signals for a small watchlist
# Minimal dependencies: streamlit, pandas, numpy, yfinance, plotly

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# ---------------- Page config (first Streamlit call) ----------------
st.set_page_config(page_title="Robust Trend & Signals", layout="wide")
st.title("📊 Robust Trend & Signals — (NIFTY & selected stocks)")

# ---------------- Symbol list (fallback set you requested) ----------------
SYMBOLS = {
    "NIFTY": "^NSEI",
    "RELIANCE": "RELIANCE.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "SBIN": "SBIN.NS",
    "TCS": "TCS.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "HDFC": "HDFC.NS",
    "BEL": "BEL.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "HAL": "HAL.NS",
    "INFOSYS": "INFY.NS",
    "VEDL": "VEDL.NS"
}

# ---------------- UI controls ----------------
left, right = st.columns([2,1])
with left:
    symbol_name = st.selectbox("Select symbol", list(SYMBOLS.keys()), index=0)
    interval = st.selectbox("Interval", ["5m", "15m", "30m", "60m", "1d"], index=1)
    chart_type = st.selectbox("Chart type", ["Candlestick", "Line"], index=0)
with right:
    st.markdown("**Settings**")
    st.write("Supertrend: period=10, multiplier=3 (safe defaults)")
    st_autorefresh = st.button("Refresh Now")

# Map interval -> default fetch period for yfinance (practical defaults)
INTERVAL_TO_PERIOD = {
    "5m": "7d",
    "15m": "15d",
    "30m": "30d",
    "60m": "60d",
    "1d": "2y"
}
default_period = INTERVAL_TO_PERIOD.get(interval, "30d")
# user can override period if they want
period = st.selectbox("Fetch period (yfinance)", [default_period, "7d", "15d", "30d", "60d", "6mo", "1y", "2y"], index=0)

# ---------------- Safe data fetcher ----------------
@st.cache_data(ttl=30)
def fetch_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetches OHLCV using yfinance and returns a cleaned DataFrame with:
    Date (datetime index), Open, High, Low, Close, Volume (numeric).
    If fetch fails or data empty, returns empty DataFrame.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure expected columns exist
    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in expected_cols):
        # try resetting and selecting columns if names differ
        df = df.reset_index()
        # prefer columns by case-insensitive match
        cols = {c.lower(): c for c in df.columns}
        mapping = {}
        for need in expected_cols:
            k = need.lower()
            if k in cols:
                mapping[cols[k]] = need
        if mapping:
            df = df.rename(columns=mapping)
        # now attempt to select expected cols
    # Keep only expected cols and drop rows with NaN close
    try:
        df = df[expected_cols].dropna(subset=["Close"])
    except Exception:
        # final safe check
        df = df.dropna(subset=df.columns.intersection(expected_cols))
        if df.empty:
            return pd.DataFrame()
        # try to ensure columns exist
        for c in expected_cols:
            if c not in df.columns:
                df[c] = np.nan

    # convert numeric
    df[["Open","High","Low","Close","Volume"]] = df[["Open","High","Low","Close","Volume"]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Close"])  # must have close
    if df.empty:
        return pd.DataFrame()

    # ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Datetime" in df.columns:
            df = df.set_index("Datetime")
        elif "Date" in df.columns:
            df = df.set_index("Date")
        else:
            df.index = pd.to_datetime(df.index)
    df.index = pd.to_datetime(df.index)
    return df

# ---------------- Indicator utilities (robust) ----------------
def compute_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    cum_pv = pv.cumsum()
    cum_vol = df["Volume"].cumsum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)
    return vwap

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # Wilder smoothing
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr

def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """
    Returns tuple (st_series, trend_series) aligned with df.index
    supertrend line (st) and trend (+1 up / -1 down)
    Implemented with numpy arrays and returns pandas Series (safe alignment).
    """
    if df.empty or len(df) < max(period + 2, 5):
        # not enough data
        return pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=int)

    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)

    atr = compute_atr(df, period=period).to_numpy(dtype=float)
    hl2 = (high + low) / 2.0
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    # initialize final bands arrays
    final_upper = upper.copy()
    final_lower = lower.copy()
    n = len(df)

    for i in range(1, n):
        # final upper
        if (upper[i] < final_upper[i - 1]) or (close[i - 1] > final_upper[i - 1]):
            final_upper[i] = upper[i]
        else:
            final_upper[i] = final_upper[i - 1]
        # final lower
        if (lower[i] > final_lower[i - 1]) or (close[i - 1] < final_lower[i - 1]):
            final_lower[i] = lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

    trend = np.ones(n, dtype=int)
    st_line = np.zeros(n, dtype=float)

    for i in range(n):
        if i == 0:
            trend[i] = 1
            st_line[i] = final_lower[i]
            continue
        if close[i] > final_upper[i - 1]:
            trend[i] = 1
        elif close[i] < final_lower[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

        st_line[i] = final_lower[i] if trend[i] == 1 else final_upper[i]

    # convert back to Series aligned with df.index
    st_series = pd.Series(st_line, index=df.index)
    trend_series = pd.Series(trend, index=df.index)
    return st_series, trend_series

# ---------------- Safe indicator pipeline ----------------
def add_indicators(df: pd.DataFrame):
    """
    Adds VWAP, EMA9, EMA21, RSI, ATR, Supertrend, Trend to df (in place copy).
    Returns df_copy (a new DataFrame).
    """
    df = df.copy()
    # Safeguard: require numeric columns
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop rows with no valid Close
    df = df.dropna(subset=["Close"])
    if df.empty:
        return df

    # VWAP
    df["VWAP"] = compute_vwap(df)

    # EMAs
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()

    # RSI
    df["RSI"] = compute_rsi(df["Close"], period=14)

    # ATR
    df["ATR"] = compute_atr(df, period=14)

    # Supertrend
    st_series, trend_series = compute_supertrend(df, period=10, multiplier=3.0)
    df["ST"] = st_series
    df["Trend"] = trend_series

    return df

# ---------------- Signal logic ----------------
def compute_signal_from_row(row):
    """Simple rule: SUPERtrend + VWAP + EMA crossover + RSI + volume filter"""
    try:
        if pd.isna(row["Trend"]):
            return "NO DATA"
        t = int(row["Trend"])
        price = float(row["Close"])
        vwap = float(row["VWAP"]) if not pd.isna(row["VWAP"]) else np.nan
        ema9 = float(row["EMA9"])
        ema21 = float(row["EMA21"])
        rsi = float(row["RSI"]) if not pd.isna(row["RSI"]) else np.nan

        if (t == 1) and (price > vwap) and (ema9 > ema21) and (rsi > 50):
            return "LONG"
        if (t == -1) and (price < vwap) and (ema9 < ema21) and (rsi < 50):
            return "SHORT"
        return "NO TRADE"
    except Exception:
        return "NO DATA"

# ---------------- Fetch + compute for selected symbol ----------------
ticker = SYMBOLS[symbol_name]
with st.spinner(f"Fetching {symbol_name} ({ticker}) ..."):
    raw = fetch_ohlcv(ticker, period=period, interval=interval)

if raw.empty:
    st.error("No data returned. Try changing Interval / Period or check network.")
    st.stop()

# compute indicators
try:
    df = add_indicators(raw)
except Exception as e:
    st.error(f"Indicator computation failed: {e}")
    st.stop()

if df.empty:
    st.error("No usable data after indicator calculations.")
    st.stop()

# compute latest signal
latest_row = df.iloc[-1]
latest_signal = compute_signal_from_row(latest_row)

# ---------- UI: top metrics ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Symbol", symbol_name)
col2.metric("Last Price", f"{latest_row['Close']:.2f}")
col3.metric("Signal", latest_signal)
col4.metric("RSI", f"{latest_row['RSI']:.2f}" if not pd.isna(latest_row["RSI"]) else "—")

st.markdown(f"**Rule reason:** Supertrend + VWAP + EMA9>EMA21 + RSI filter (simple heuristic).")

# ---------- Chart (plotly) ----------
st.subheader("Price chart")
fig = go.Figure()
if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))
else:
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))

# overlays
fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], mode="lines", name="VWAP", line=dict(width=1)))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], mode="lines", name="EMA9", line=dict(width=1)))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], mode="lines", name="EMA21", line=dict(width=1)))
fig.add_trace(go.Scatter(x=df.index, y=df["ST"], mode="lines", name="Supertrend", line=dict(width=1)))

# mark signal dots (where trend changed or buy/short conditions)
sig_mask = df.apply(lambda r: compute_signal_from_row(r) in ("LONG","SHORT"), axis=1)
sig_points = df[sig_mask]
if not sig_points.empty:
    fig.add_trace(go.Scatter(
        x=sig_points.index, y=sig_points["Close"], mode="markers",
        marker=dict(size=8, color=[ "green" if compute_signal_from_row(r)=="LONG" else "red" for _,r in sig_points.iterrows() ]),
        name="Signal Points"
    ))

fig.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ---------- Show last rows ----------
st.subheader("Latest data (last 20 rows)")
display_cols = ["Close","VWAP","EMA9","EMA21","RSI","ATR","ST","Trend"]
st.dataframe(df[display_cols].tail(20).assign(Date=lambda d: d.index.strftime("%Y-%m-%d %H:%M")))

# ---------- Quick watchlist scan for all symbols ----------
st.subheader("Quick scan — watchlist")
scan = []
for name, tk in SYMBOLS.items():
    try:
        d = fetch_ohlcv(tk, period=INTERVAL_TO_PERIOD.get(interval, "30d"), interval=interval)
        if d.empty:
            scan.append({"Symbol": name, "Price": None, "Signal": "NO DATA"})
            continue
        d2 = add_indicators(d)
        if d2.empty:
            scan.append({"Symbol": name, "Price": None, "Signal": "NO DATA"})
            continue
        last = d2.iloc[-1]
        scan.append({"Symbol": name, "Price": float(last["Close"]), "Signal": compute_signal_from_row(last)})
    except Exception as e:
        scan.append({"Symbol": name, "Price": None, "Signal": "ERR"})
scan_df = pd.DataFrame(scan)
st.dataframe(scan_df)

# ---------- Download CSV ----------
st.download_button("Download scan CSV", scan_df.to_csv(index=False).encode(), file_name=f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

st.caption("This app is decision-support only. Signals are rule-of-thumb and require backtesting and risk controls before live use.")
