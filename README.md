# NSE Options Live Dashboard

A Streamlit dashboard that displays live (best-effort) metrics for Indian index options:
- India VIX (trend)
- PCR (Put/Call Open Interest ratio)
- ATM straddle (CE+PE) and bid–ask spreads
- ATM IV (implied volatility) via Black–Scholes
- RSI & ADX on NIFTY/BANKNIFTY intraday candles (via Yahoo Finance)

## How to run

1. Install Python 3.10+
2. Create a virtual environment (recommended)
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

> **Notes**
> - NSE may throttle/block frequent requests. Keep refresh interval >= 30s.
> - Data is for educational purposes only. Double-check values before trading.
> - If NSE option-chain endpoint changes, you may need to update `fetch_option_chain`.