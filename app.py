# app.py
# Intraday / Daily Trend + Entry/Exit signal dashboard for selected symbols
# Symbols: NIFTY (^NSEI), RELIANCE.NS, TCS.NS, KOTAKBANK.NS, SBIN.NS, TATAMOTORS.NS
# Indicators: VWAP, EMA(9,21), RSI(14), ATR, Supertrend
# Signals: rule-based Long / Short / Exit

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -------------------------
# Page config (must be first streamlit command)
# -------------------------
st.set_page_config(page_title="Trend & Signals — NIFTY / Stocks", layout="wide")

# -------------------------
# Helper functions (indicators)
# -------------------------
def vwap(df):
    """Calculate VWAP for dataframe with columns: High, Low, Close, Volume"""
    # Typical price * volume cumulative divided by volume cumulative
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"]
    cum_pv = pv.cumsum()
    cum_vol = df["Volume"].cumsum()
    return cum_pv / cum_vol

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def true_range(df):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(df, period=14):
    tr = true_range(df)
    return tr.ewm(span=period, adjust=False).mean()

def supertrend(df, period=10, multiplier=3.0):
    """
    Returns dataframe with ST line and Trend column (+1 up, -1 down).
    Implementation uses .iloc indexing to avoid pandas iat/get_value issues.
    """
    out = df.copy().reset_index(drop=True)
    hl2 = (out["High"] + out["Low"]) / 2.0
    _atr = atr(out, period=period)
    upperband = hl2 + multiplier * _atr
    lowerband = hl2 - multiplier * _atr

    final_upper = upperband.copy()
    final_lower = lowerband.copy()

    # Initialize final bands
    for i in range(1, len(out)):
        # final upper
        if (upperband.iloc[i] < final_upper.iloc[i - 1]) or (out["Close"].iloc[i - 1] > final_upper.iloc[i - 1]):
            final_upper.iloc[i] = upperband.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]
        # final lower
        if (lowerband.iloc[i] > final_lower.iloc[i - 1]) or (out["Close"].iloc[i - 1] < final_lower.iloc[i - 1]):
            final_lower.iloc[i] = lowerband.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

    trend = np.ones(len(out), dtype=int)
    st_line = np.zeros(len(out))

    for i in range(len(out)):
        if i == 0:
            trend[i] = 1
            st_line[i] = final_lower.iloc[i]
        else:
            if out["Close"].iloc[i] > final_upper.iloc[i - 1]:
                trend[i] = 1
            elif out["Close"].iloc[i] < final_lower.iloc[i - 1]:
                trend[i] = -1
            else:
                trend[i] = trend[i - 1]
            st_line[i] = final_lower.iloc[i] if trend[i] == 1 else final_upper.iloc[i]

    out["ST"] = st_line
    out["Trend"] = trend
    return out.set_index(df.index)

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

# -------------------------
# Signal logic (rule-based)
# -------------------------
def compute_signals(df):
    """
    Input: df with OHLCV index and computed indicators:
      - VWAP, EMA9, EMA21, RSI, Trend (Supertrend)
    Returns: last row signal (Long/Short/Exit/No Trade) and supporting reason text
    """
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    price = last["Close"]
    vwap_val = last["VWAP"]
    ema9 = last["EMA9"]
    ema21 = last["EMA21"]
    st_trend = last["Trend"]
    rsi_val = last["RSI"]
    vol = last["Volume"]
    vol_avg = df["Volume"].rolling(20, min_periods=1).mean().iloc[-1]

    # Conditions
    bullish = (st_trend == 1) and (price > vwap_val) and (ema9 > ema21) and (rsi_val > 50) and (vol >= vol_avg * 0.6)
    bearish = (st_trend == -1) and (price < vwap_val) and (ema9 < ema21) and (rsi_val < 50) and (vol >= vol_avg * 0.6)

    # Entry/Exit rules
    if bullish:
        entry = True
        reason = "Supertrend Up + Price>VWAP + EMA9>EMA21 + RSI>50"
        action = "LONG"
    elif bearish:
        entry = True
        reason = "Supertrend Down + Price<VWAP + EMA9<EMA21 + RSI<50"
        action = "SHORT"
    else:
        # check if an existing position should be exited (trend flip or momentum loss)
        if prev["Trend"] == 1 and st_trend == -1:
            action = "EXIT LONG"
            reason = "Supertrend flipped to Down"
        elif prev["Trend"] == -1 and st_trend == 1:
            action = "EXIT SHORT"
            reason = "Supertrend flipped to Up"
        elif (prev["Close"] > prev["VWAP"]) and (last["Close"] < last["VWAP"]):
            action = "EXIT LONG"
            reason = "Price crossed below VWAP"
        elif (prev["Close"] < prev["VWAP"]) and (last["Close"] > last["VWAP"]):
            action = "EXIT SHORT"
            reason = "Price crossed above VWAP"
        else:
            action = "NO TRADE"
            reason = "No clear confluence"

    # Risk notes / suggested SL targets (simple heuristics)
    # SL for long: recent ATR * 1.5 below entry; for short: above entry
    at = df["ATR"].iloc[-1] if "ATR" in df.columns else np.nan
    sl_long = price - (1.5 * at) if not np.isnan(at) else np.nan
    sl_short = price + (1.5 * at) if not np.isnan(at) else np.nan

    return {
        "action": action,
        "reason": reason,
        "price": price,
        "vwap": vwap_val,
        "ema9": ema9,
        "ema21": ema21,
        "rsi": rsi_val,
        "volume": vol,
        "vol_avg": vol_avg,
        "atr": at,
        "sl_long": sl_long,
        "sl_short": sl_short
    }

# -------------------------
# UI controls
# -------------------------
st.sidebar.header("Settings")
symbols_map = {
    "NIFTY": "^NSEI",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "SBIN": "SBIN.NS",
    "TATAMOTORS": "TATAMOTORS.NS"
}
symbol = st.sidebar.selectbox("Symbol", list(symbols_map.keys()), index=0)
timeframe = st.sidebar.selectbox("Timeframe", ["5m", "15m", "60m", "1d"], index=1)
lookback_period = st.sidebar.slider("Lookback bars (for indicators)", min_value=50, max_value=1000, value=400, step=50)
st.sidebar.markdown("Signals use Supertrend + VWAP + EMA crossover + RSI + volume confirmation.")

if st.sidebar.button("Refresh Now"):
    st.experimental_rerun()

# -------------------------
# Fetch data (yfinance)
# -------------------------
yf_sym = symbols_map[symbol]
# map timeframe to yfinance interval and period to request enough history
# For intraday intervals yfinance needs periods like '5d' etc.
if timeframe == "5m":
    interval = "5m"
    period = "10d"
elif timeframe == "15m":
    interval = "15m"
    period = "60d"  # more days to get enough bars
elif timeframe == "60m":
    interval = "60m"
    period = "120d"
else:
    interval = "1d"
    period = "2y"

@st.cache_data(ttl=60)
def load_ohlcv(ticker, period, interval):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if df.empty:
            return df
        df = df.reset_index()
        # rename index column as Date/Datetime
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        elif 'Date' in df.columns:
            pass
        # Ensure columns 'Open','High','Low','Close','Volume' exist
        df = df[['Date','Open','High','Low','Close','Volume']]
        return df
    except Exception as e:
        st.error(f"Price fetch error: {e}")
        return pd.DataFrame()

with st.spinner(f"Fetching {symbol} {interval} data from Yahoo Finance..."):
    price_df = load_ohlcv(yf_sym, period, interval)

if price_df.empty:
    st.error("No price data available. Try another timeframe or symbol.")
    st.stop()

# trim to lookback
if len(price_df) > lookback_period:
    df_calc = price_df.tail(lookback_period).copy()
else:
    df_calc = price_df.copy()

# set index as Date for plotting convenience
df_calc.set_index("Date", inplace=True)

# -------------------------
# Compute indicators
# -------------------------
df_calc["VWAP"] = vwap(df_calc)
df_calc["EMA9"] = ema(df_calc["Close"], 9)
df_calc["EMA21"] = ema(df_calc["Close"], 21)
df_calc["RSI"] = rsi(df_calc["Close"], 14)
df_calc["ATR"] = atr(df_calc, period=14)
# Supertrend needs numeric index preserved; create using copy of df_calc
df_st = supertrend(df_calc[["Open","High","Low","Close","Volume"]], period=10, multiplier=3.0)
# df_st returned has same index as input; assign Trend and ST
df_calc["ST"] = df_st["ST"]
df_calc["Trend"] = df_st["Trend"]

# -------------------------
# Generate signal for this symbol (based on latest bars)
# -------------------------
signal = compute_signals(df_calc)

# -------------------------
# Display top metrics & signal summary
# -------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Symbol", symbol)
col2.metric("Last Price", f"{signal['price']:.2f}")
col3.metric("Signal", signal["action"])
col4.metric("RSI", f"{signal['rsi']:.2f}")

st.markdown(f"**Reason:** {signal['reason']}")
if signal["action"] in ["LONG", "SHORT"]:
    st.markdown(f"Suggested stop-loss (approx): {'{:.2f}'.format(signal['sl_long']) if signal['action']=='LONG' else '{:.2f}'.format(signal['sl_short'])}")

st.write("---")

# -------------------------
# Price chart with overlays (Plotly)
# -------------------------
st.subheader(f"{symbol} Price Chart ({timeframe})")
fig = go.Figure()

# Candlestick
fig.add_trace(go.Candlestick(
    x=df_calc.index,
    open=df_calc["Open"],
    high=df_calc["High"],
    low=df_calc["Low"],
    close=df_calc["Close"],
    name="Price"
))

# VWAP
fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc["VWAP"], mode="lines", name="VWAP", line=dict(color="cyan")))

# EMA9 & EMA21
fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc["EMA9"], mode="lines", name="EMA9", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc["EMA21"], mode="lines", name="EMA21", line=dict(color="magenta")))

# Supertrend line
fig.add_trace(go.Scatter(x=df_calc.index, y=df_calc["ST"], mode="lines", name="Supertrend", line=dict(color="yellow")))

# Last signal marker
last_time = df_calc.index[-1]
fig.add_trace(go.Scatter(x=[last_time], y=[signal["price"]], mode="markers+text",
                         marker=dict(size=12, color="green" if signal["action"]=="LONG" else "red" if signal["action"]=="SHORT" else "gray"),
                         text=[signal["action"]],
                         textposition="top center",
                         name="Signal"))

fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Show indicator table (last few rows)
# -------------------------
st.subheader("Last 10 rows — indicators")
st.dataframe(df_calc[["Open","High","Low","Close","Volume","VWAP","EMA9","EMA21","RSI","ATR","Trend"]].tail(10))

# -------------------------
# Quick multi-symbol scan (summary table)
# -------------------------
st.subheader("Quick Scan — All Symbols")
summary_rows = []
for sym in symbols_map.keys():
    try:
        df_tmp = load_ohlcv(symbols_map[sym], period="30d" if interval.endswith("m") else "1y", interval)
        if df_tmp.empty:
            summary_rows.append({"Symbol": sym, "Signal": "NO DATA", "Price": np.nan, "Reason": ""})
            continue
        # compute small set of indicators on last 200 bars
        if len(df_tmp) > 200:
            d = df_tmp.tail(200).reset_index().set_index("Datetime" if "Datetime" in df_tmp.columns else "Date")
        else:
            d = df_tmp.reset_index().set_index("Datetime" if "Datetime" in df_tmp.columns else "Date")
        d = d.rename_axis("Date")
        d = d[["Open","High","Low","Close","Volume"]]
        d["VWAP"] = vwap(d)
        d["EMA9"] = ema(d["Close"], 9)
        d["EMA21"] = ema(d["Close"], 21)
        d["RSI"] = rsi(d["Close"], 14)
        d["ATR"] = atr(d, 14)
        st_data = supertrend(d[["Open","High","Low","Close","Volume"]], period=10, multiplier=3.0)
        d["Trend"] = st_data["Trend"]
        sig = compute_signals(d)
        summary_rows.append({"Symbol": sym, "Signal": sig["action"], "Price": sig["price"], "Reason": sig["reason"]})
    except Exception as e:
        summary_rows.append({"Symbol": sym, "Signal": "ERR", "Price": np.nan, "Reason": str(e)})

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df)

# -------------------------
# Download signals CSV
# -------------------------
csv = summary_df.to_csv(index=False).encode()
st.download_button("Download scan CSV", csv, file_name=f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")

st.caption("This tool is decision-support only. Test and paper-trade strategies before using real capital. Signals are rule-based combining Supertrend, VWAP, EMA crossover, RSI and volume confirmation.")
