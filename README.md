# Intraday Quant Trading Bot

This project implements an intraday trading signal engine for Indian equities. It dynamically selects stocks and generates long/short signals using a multi-indicator confluence model.

## Features

- Dynamically selects stocks from NIFTY50, NIFTY NEXT 50, MIDCAP, and SMALLCAP universes using opening-range breakout and consolidation-breakout scans.
- Uses NSE India as primary source for dynamic stock change analysis, with yfinance fallback.
- Fetches intraday 1-minute OHLCV data.
- Computes multi-factor indicators: RSI, MACD, EMA20/EMA50, ADX, Bollinger Bands, ATR, VWAP, Volume SMA, OBV.
- Generates long/short entry-exit signals with confidence scores.
- Enforces portfolio risk controls: max 20% capital per trade, 1% risk per trade, 2.5% max daily loss stop, and max open-trade cap.
- Includes market-time and NSE holiday awareness (IST).
- Provides Streamlit dashboard with simulated long/short PnL tracking.
- Provides post-market report generation with signal and trade CSV exports.

## Requirements

- Python 3.8+
- Libraries listed in requirements.txt

## Installation

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the script:
```
python main.py
```

Generate post-market report (after 3:30 PM IST):
```
python market_close_report.py
```

Optional:
```
python market_close_report.py --output-dir reports
python market_close_report.py --force
```

For the interactive dashboard:
```
streamlit run dashboard.py
```

The dashboard displays actionable signals, confidence, position state, and simulated long/short PnL.

## Strategy

The strategy uses weighted confluence rather than RSI/MACD alone.

### Indicators Used
- Momentum: RSI, MACD crossover
- Trend: EMA20 vs EMA50, price vs VWAP
- Trend strength: ADX
- Volatility: Bollinger Band context, ATR%
- Volume/flow: Volume spike vs SMA20, OBV slope

### Signal States
- `BUY`: Open long position
- `SHORT`: Open short position
- `EXIT_LONG`: Close long position
- `COVER`: Close short position
- `HOLD`: No action

If shorting is disabled in code, bearish actions are mapped to `SELL`.

### Entry/Exit Logic (high level)
- Entries require MACD crossover plus minimum confluence score.
- Exits are intentionally easier than entries to protect intraday PnL.
- Each signal carries a confidence score (`Signal_Confidence`).
- Stock selection prioritizes either:
  - Opening Range Breakout (ORB), or
  - Consolidation range breakout.

### Post-Market Report
- Run after 3:30 PM IST:
  - `python market_close_report.py`
- Includes:
  - Per-signal event details (stock, signal, time, price, confidence)
  - Trade log (entry/exit, side, PnL %)
  - Day PnL % summary

## Disclaimer

This is for educational purposes only. Trading involves risk. Always do your own research and consult professionals.
