# app.py
# Robust Trend & Signal Dashboard for selected instruments (NIFTY + Stocks)
# Minimal dependencies: streamlit, pandas, numpy, yfinance, plotly
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math

# --------- Page config (first Streamlit call) ----------
st.set_page_config(page_title="Trend & Signals — NIFTY + Stocks", layout="wide")
st.title("📊 Trend & Signals — NIFTY, Reliance, TCS, Kotak, SBI, Tata Motors")

# --------- Symbols map ----------
SYMBOLS = {
    "NIFTY": "^NSEI",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "SBIN": "SBIN.NS",
    "TATAMOTORS": "TATAMOTORS.NS"
}

# --------- Sidebar controls ----------
st.sidebar.header("Settings")
symbol_name = st.sidebar.selectbox("Symbol", list(SYMBOLS.keys()), index=0)
interval = st.sidebar.selectbox("Interval", ["5m", "15m", "60m", "1d"], index=1)
period = st.sidebar.selectbox("Period (yfinance)", ["5d", "15d", "1mo", "3mo", "6mo", "1y"], index=1)
lookback = st.sidebar.slider("Indicator lookback bars", 100, 2000, 500, step=50)
min_vol_mult = st.sidebar.slider("Min volume multiplier (confirmation)", 0.4, 2.0, 0.6, step=0.1)

if st.sidebar.button("Refresh now"):
    st.experimental_rerun()

# --------- Helper indicator functions (vectorized / robust) ----------

def compute_vwap(df):
    # df must have High, Low, Close, Volume
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    cum_pv = pv.cumsum()
    cum_vol = df["Volume"].cumsum()
    vwap = cum_pv / np.where(cum_vol == 0, np.nan, cum_vol)
    return vwap

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Wilder smoothing via ewm with alpha=1/period
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / np.where(ma_down == 0, np.nan, ma_down)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(df, period=14):
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()
    # True range array
    tr = np.zeros(len(df))
    for i in range(len(df)):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    # Wilder ATR: use ewm with alpha=1/period on TR series
    tr_series = pd.Series(tr)
    atr = tr_series.ewm(alpha=1/period, adjust=False).mean().to_numpy()
    return atr

def supertrend_numpy(df, period=10, multiplier=3.0):
    """
    Robust supertrend using numpy arrays to avoid pandas indexing pitfalls.
    Input df must have columns Open, High, Low, Close (pandas DataFrame).
    Returns pandas Series ST and Trend aligned with df index.
    """
    if len(df) < period + 2:
        # not enough data
        return pd.Series([np.nan]*len(df), index=df.index), pd.Series([0]*len(df), index=df.index)

    hl2 = ((df["High"] + df["Low"]) / 2.0).to_numpy()
    atr = compute_atr(df, period=period)  # numpy array
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    final_upper = upper.copy()
    final_lower = lower.copy()

    close = df["Close"].to_numpy()
    n = len(df)

    # Build final bands
    for i in range(1, n):
        if close[i-1] <= final_upper[i-1]:
            final_upper[i] = min(upper[i], final_upper[i-1])
        else:
            final_upper[i] = upper[i]

        if close[i-1] >= final_lower[i-1]:
            final_lower[i] = max(lower[i], final_lower[i-1])
        else:
            final_lower[i] = lower[i]

    # trend and stline
    trend = np.ones(n, dtype=int)
    stline = np.zeros(n, dtype=float)

    for i in range(n):
        if i == 0:
            trend[i] = 1
            stline[i] = final_lower[i]
            continue
        if close[i] > final_upper[i-1]:
            trend[i] = 1
        elif close[i] < final_lower[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

        stline[i] = final_lower[i] if trend[i] == 1 else final_upper[i]

    return pd.Series(stline, index=df.index), pd.Series(trend, index=df.index)

# --------- Signal logic (rule-based) ----------
def generate_signal(df, vol_mult=0.6):
    """
    Expects df with columns: Close, VWAP, EMA9, EMA21, RSI, ST (supertrend line), Trend, ATR, Volume
    Returns a dict with action and details.
    """
    if len(df) < 2:
        return {"action":"NO DATA", "reason":"insufficient bars"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(last["Close"])
    vwap = float(last["VWAP"]) if not pd.isna(last["VWAP"]) else np.nan
    ema9 = float(last["EMA9"]) if not pd.isna(last["EMA9"]) else np.nan
    ema21 = float(last["EMA21"]) if not pd.isna(last["EMA21"]) else np.nan
    rsi = float(last["RSI"]) if not pd.isna(last["RSI"]) else np.nan
    trend = int(last["Trend"]) if "Trend" in df.columns else 0
    atr = float(last["ATR"]) if not pd.isna(last["ATR"]) else np.nan
    vol = float(last["Volume"])
    vol_avg = float(df["Volume"].tail(50).mean()) if len(df) >= 5 else float(df["Volume"].mean())

    # conditions
    vol_ok = vol >= (vol_avg * vol_mult)
    bullish = (trend == 1) and (price > vwap) and (ema9 > ema21) and (rsi > 50) and vol_ok
    bearish = (trend == -1) and (price < vwap) and (ema9 < ema21) and (rsi < 50) and vol_ok

    if bullish:
        action = "LONG"
        reason = "Supertrend Up + Price>VWAP + EMA9>EMA21 + RSI>50 + Vol confirmation"
        sl = price - 1.5 * atr if not math.isnan(atr) else None
    elif bearish:
        action = "SHORT"
        reason = "Supertrend Down + Price<VWAP + EMA9<EMA21 + RSI<50 + Vol confirmation"
        sl = price + 1.5 * atr if not math.isnan(atr) else None
    else:
        # Check exit conditions (trend flip or VWAP cross)
        if prev.get("Trend", 0) == 1 and trend == -1:
            action = "EXIT LONG"
            reason = "Supertrend flipped to down"
            sl = None
        elif prev.get("Trend", 0) == -1 and trend == 1:
            action = "EXIT SHORT"
            reason = "Supertrend flipped to up"
            sl = None
        elif prev["Close"] > prev["VWAP"] and last["Close"] < last["VWAP"]:
            action = "EXIT LONG"
            reason = "Price crossed below VWAP"
            sl = None
        elif prev["Close"] < prev["VWAP"] and last["Close"] > last["VWAP"]:
            action = "EXIT SHORT"
            reason = "Price crossed above VWAP"
            sl = None
        else:
            action = "NO TRADE"
            reason = "No clear confluence"
            sl = None

    return {
        "action": action,
        "reason": reason,
        "price": price,
        "sl": sl,
        "rsi": rsi,
        "atr": atr,
        "volume": vol,
        "vol_avg": vol_avg
    }

# --------- Data fetcher with safe handling ----------
@st.cache_data(ttl=60)
def fetch_ohlcv(ticker, period, interval):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        # unify datetime column name
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime":"Date"})
        # keep required columns and drop rows missing essential values
        df = df[["Date","Open","High","Low","Close","Volume"]].dropna()
        return df
    except Exception as e:
        st.error(f"Price fetch error: {e}")
        return pd.DataFrame()

# --------- Fetch data and compute indicators ----------
yf_ticker = SYMBOLS[symbol_name]
with st.spinner(f"Fetching {symbol_name} ({yf_ticker}) data..."):
    df = fetch_ohlcv(yf_ticker, period=period, interval=interval)

if df.empty:
    st.error("No price data returned — try different period/interval or check your network.")
    st.stop()

# possibly trim to lookback most recent bars
if len(df) > lookback:
    df = df.tail(lookback).reset_index(drop=True)

# Compute indicators (vectorized)
df["VWAP"] = compute_vwap(df)
df["EMA9"] = ema(df["Close"], 9)
df["EMA21"] = ema(df["Close"], 21)
df["RSI"] = compute_rsi(df["Close"], 14)
df["ATR"] = compute_atr(df, period=14)

stline, trend = supertrend_numpy(df, period=10, multiplier=3.0)
df["ST"] = stline
df["Trend"] = trend

# ensure types
df["Trend"] = df["Trend"].astype(int)

# --------- Compute latest signal ----------
sig = generate_signal(df, vol_mult=min_vol_mult)

# --------- UI: Top metrics ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Symbol", symbol_name)
col2.metric("Last Price", f"{sig['price']:.2f}")
col3.metric("Signal", sig["action"])
col4.metric("RSI", f"{sig['rsi']:.2f}" if sig["rsi"] is not None else "—")
st.markdown(f"**Reason:** {sig['reason']}")
if sig["sl"] is not None:
    st.markdown(f"Suggested SL (approx): {sig['sl']:.2f}")

st.markdown("---")

# --------- Price chart (Plotly) ----------
st.subheader(f"{symbol_name} price chart ({interval})")
fig = go.Figure()
time_col = df["Date"]

# Candles / line
fig.add_trace(go.Candlestick(
    x=time_col, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
))
# VWAP
fig.add_trace(go.Scatter(x=time_col, y=df["VWAP"], mode="lines", name="VWAP", line=dict(color="cyan")))
# EMA9 & EMA21
fig.add_trace(go.Scatter(x=time_col, y=df["EMA9"], mode="lines", name="EMA9", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=time_col, y=df["EMA21"], mode="lines", name="EMA21", line=dict(color="magenta")))
# Supertrend
fig.add_trace(go.Scatter(x=time_col, y=df["ST"], mode="lines", name="Supertrend", line=dict(color="yellow")))

# mark latest signal point
color_map = {"LONG":"green","SHORT":"red","EXIT LONG":"orange","EXIT SHORT":"orange","NO TRADE":"gray","NO DATA":"gray"}
marker_color = color_map.get(sig["action"], "gray")
fig.add_trace(go.Scatter(x=[df["Date"].iloc[-1]], y=[sig["price"]], mode="markers+text",
                         marker=dict(size=14, color=marker_color), text=[sig["action"]], textposition="top center", name="Signal"))

fig.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# --------- Recent indicators table ----------
st.subheader("Latest indicator rows")
disp_cols = ["Date","Open","High","Low","Close","Volume","VWAP","EMA9","EMA21","RSI","ATR","ST","Trend"]
st.dataframe(df[disp_cols].tail(20).assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")))

# --------- Quick scan: compute simple signal across all symbols (fast) ----------
st.subheader("Quick Scan — all symbols")
scan_rows = []
for name, ticker in SYMBOLS.items():
    try:
        tmp = fetch_ohlcv(ticker, period="30d" if interval.endswith("m") else "1y", interval=interval)
        if tmp.empty:
            scan_rows.append({"Symbol": name, "Signal":"NO DATA", "Price": None, "Reason": ""})
            continue
        # keep small recent window
        if len(tmp) > 300:
            tmp = tmp.tail(300).reset_index(drop=True)
        # compute indicators quickly
        tmp["VWAP"] = compute_vwap(tmp)
        tmp["EMA9"] = ema(tmp["Close"], 9)
        tmp["EMA21"] = ema(tmp["Close"], 21)
        tmp["RSI"] = compute_rsi(tmp["Close"], 14)
        tmp["ATR"] = compute_atr(tmp, 14)
        stline_tmp, trend_tmp = supertrend_numpy(tmp, period=10, multiplier=3.0)
        tmp["ST"] = stline_tmp
        tmp["Trend"] = trend_tmp.astype(int)
        s = generate_signal(tmp, vol_mult=min_vol_mult)
        scan_rows.append({"Symbol": name, "Signal": s["action"], "Price": s["price"], "Reason": s["reason"]})
    except Exception as e:
        scan_rows.append({"Symbol": name, "Signal":"ERR", "Price": None, "Reason": str(e)})

scan_df = pd.DataFrame(scan_rows)
st.dataframe(scan_df)

# --------- Download scan results ----------
csv_bytes = scan_df.to_csv(index=False).encode()
st.download_button("Download scan CSV", csv_bytes, file_name=f"signals_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")

st.caption("This dashboard provides rule-based signals for research and decision support only. Backtest and paper-trade before using real capital. Indicators use best-effort calculations; adjust parameters to suit your style.")
