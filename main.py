"""
main.py  –  Intraday quant engine for NSE equities
Optimizations applied:
  1.  Kite Connect as primary live data feed (yfinance as fallback)
  2.  VWAP anchored to today's session only (multi-day contamination fix)
  3.  Transaction cost model (brokerage + STT + exchange + stamp duty)
  4.  MACD(6,13,5) tuned for 1-minute bars
  5.  RSI(7) for intraday responsiveness
  6.  EMA34 replaces EMA50 (faster trend filter)
  7.  Volume spike raised to 1.5× (reduce false confluence)
  8.  Supertrend(7, 3.0) added to confluence scoring (score_max → 13)
  9.  ORB opening range uses time-based filtering (gap-safe)
  10. Cache-bust key exported for dashboard use
"""

import datetime
import concurrent.futures
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FALLBACK_STOCKS = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
IST_ZONE = ZoneInfo("Asia/Kolkata")
NSE_HOLIDAY_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
ENABLE_SHORTING = False
LONG_ONLY_TOP_GAINERS = 5

# Portfolio / risk
STARTING_CAPITAL = 100_000.0
MAX_CAPITAL_PER_TRADE_PCT = 0.20
RISK_PER_TRADE_PCT = 0.01
DAILY_MAX_LOSS_PCT = 0.025
MAX_OPEN_TRADES = 3
STOP_LOSS_ATR_MULTIPLIER = 1.0
MIN_STOP_LOSS_PCT = 0.003
MIN_ENTRY_CONFIDENCE = 56.0

# Transaction cost model (NSE intraday equity, Zerodha-style)
# Brokerage: ₹20 flat per leg or 0.03% whichever is lower → approximate as 0.03%
BROKERAGE_PCT = 0.0003          # per leg; × 2 for round trip
STT_PCT = 0.00025               # on sell-side only (intraday equity)
EXCHANGE_CHARGES_PCT = 0.0000345  # NSE turnover charge, per leg
SEBI_CHARGES_PCT = 0.000001     # per leg
STAMP_DUTY_PCT = 0.00003        # on buy-side only
# Combined round-trip cost as fraction of capital_used
ROUND_TRIP_COST_PCT = (
    BROKERAGE_PCT * 2
    + STT_PCT
    + EXCHANGE_CHARGES_PCT * 2
    + SEBI_CHARGES_PCT * 2
    + STAMP_DUTY_PCT
)

# Breakout universe selection
BREAKOUT_UNIVERSE_SIZE = 10
OPENING_RANGE_MINUTES = 15      # first N minutes after 09:15 IST
CONSOLIDATION_LOOKBACK_BARS = 20
CONSOLIDATION_RANGE_PCT_MAX = 1.2
BREAKOUT_CONFIRM_BARS = 2
MIN_DAILY_VOLUME = 500_000
MIN_DAILY_TRADED_VALUE = 200_000_000.0
MIN_TRADABLE_PRICE = 50.0
LIVE_DATA_MAX_CANDLE_AGE_MINUTES = 20
DATA_FETCH_RETRIES = 3
DATA_FETCH_RETRY_SECONDS = 1.0

# ---------------------------------------------------------------------------
# Kite Connect configuration
# Populate KITE_API_KEY + KITE_ACCESS_TOKEN at runtime (env vars or secrets).
# When both are present, Kite is used as the primary 1-minute data feed.
# ---------------------------------------------------------------------------
import os
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "")

# NSE symbol → Kite instrument token mapping.
# Extend this dict or load from kite.instruments("NSE") at startup.
KITE_INSTRUMENT_TOKENS: dict[str, int] = {
    # Format:  "SYMBOL": token_integer
    # e.g.   "RELIANCE": 738561,
    # Populate by running:
    #   from kiteconnect import KiteConnect
    #   kite = KiteConnect(api_key=KITE_API_KEY)
    #   kite.set_access_token(KITE_ACCESS_TOKEN)
    #   instruments = kite.instruments("NSE")
    #   {i['tradingsymbol']: i['instrument_token'] for i in instruments}
}


def _get_kite_client():
    """Return a logged-in KiteConnect instance or None if unavailable."""
    if not (KITE_API_KEY and KITE_ACCESS_TOKEN):
        return None
    try:
        from kiteconnect import KiteConnect  # pip install kiteconnect
        kite = KiteConnect(api_key=KITE_API_KEY)
        kite.set_access_token(KITE_ACCESS_TOKEN)
        return kite
    except Exception:
        return None


def fetch_intraday_data_kite(ticker: str) -> pd.DataFrame:
    """
    Fetch today's 1-minute OHLCV bars from Kite Connect.
    Returns an empty DataFrame if Kite is not configured or the symbol
    token is not in KITE_INSTRUMENT_TOKENS.
    """
    kite = _get_kite_client()
    if kite is None:
        return pd.DataFrame()

    symbol = normalize_symbol(ticker)
    token = KITE_INSTRUMENT_TOKENS.get(symbol)
    if not token:
        return pd.DataFrame()

    now_ist = get_ist_now()
    from_dt = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    to_dt = now_ist

    try:
        records = kite.historical_data(token, from_dt, to_dt, "minute")
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df = df.rename(columns={
            "date": "Datetime", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume",
        })
        df = df.set_index("Datetime")
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize(IST_ZONE)
        df = df.dropna(subset=["Close"])
        return df
    except Exception as exc:
        print(f"Kite data fetch failed for {ticker}: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_symbol(stock: str) -> str:
    return stock.replace(".NS", "").strip().upper()


def to_float(value):
    try:
        if isinstance(value, str):
            value = value.replace(",", "")
        return float(value)
    except (TypeError, ValueError):
        return None


def get_ist_now() -> datetime.datetime:
    return datetime.datetime.now(tz=IST_ZONE)


def get_cache_bust_key() -> str:
    """Per-minute string used by the dashboard to invalidate stale snapshots."""
    return get_ist_now().strftime("%Y%m%d%H%M")


# ---------------------------------------------------------------------------
# NSE session / HTTP helpers
# ---------------------------------------------------------------------------
def init_nse_session() -> requests.Session | None:
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.nseindia.com/",
    })
    try:
        session.get("https://www.nseindia.com/", timeout=8)
    except requests.RequestException:
        return None
    return session


def nse_get_json(session, url, timeout=6, retries=3, backoff_seconds=0.6):
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code in (401, 403, 429, 500, 502, 503, 504):
                session.get("https://www.nseindia.com/", timeout=8)
                if attempt < retries - 1:
                    time.sleep(backoff_seconds * (2 ** attempt))
                    continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt < retries - 1:
                time.sleep(backoff_seconds * (2 ** attempt))
                continue
            return None
    return None


# ---------------------------------------------------------------------------
# Dynamic stock universe helpers
# ---------------------------------------------------------------------------
def fetch_stock_change_from_nse(stock, session):
    symbol = normalize_symbol(stock)
    url = f"https://www.nseindia.com/api/quote-equity?symbol={quote(symbol)}"
    payload = nse_get_json(session, url, timeout=5, retries=3, backoff_seconds=0.7)
    if not payload:
        return None
    price_info = payload.get("priceInfo", {})
    change = to_float(price_info.get("pChange"))
    if change is None:
        return None
    return stock, change


def fetch_stock_change_from_yfinance(stock):
    try:
        data = yf.Ticker(stock).history(period="6d", interval="1d", timeout=5)
        if data.empty or len(data) < 2:
            return None
        close_series = data["Close"]
        if isinstance(close_series, pd.DataFrame):
            if close_series.shape[1] != 1:
                return None
            close_series = close_series.iloc[:, 0]
        prev_close = float(close_series.iloc[-2])
        current = float(close_series.iloc[-1])
        if prev_close == 0:
            return None
        return stock, (current - prev_close) / prev_close * 100
    except Exception:
        return None


def get_dynamic_stocks_from_nse(all_tickers):
    session = init_nse_session()
    if session is None:
        return []
    all_changes = []
    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(fetch_stock_change_from_nse, stock, session): stock
                for stock in all_tickers
            }
            for future in concurrent.futures.as_completed(futures, timeout=35):
                result = future.result()
                if result is not None:
                    all_changes.append(result)
    except TimeoutError:
        print("NSE dynamic fetch timed out, trying fallback source.")
    except Exception as e:
        print(f"Error fetching NSE stock changes: {e}")
    finally:
        session.close()
    return all_changes


def get_dynamic_stocks_from_yfinance(all_tickers):
    all_changes = []
    try:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(fetch_stock_change_from_yfinance, stock): stock
                for stock in all_tickers
            }
            for future in concurrent.futures.as_completed(futures, timeout=30):
                result = future.result()
                if result is not None:
                    all_changes.append(result)
    except TimeoutError:
        print("yfinance dynamic fetch timed out, using fallback stock list.")
    except Exception as e:
        print(f"Error fetching yfinance stock changes: {e}")
    return all_changes


# ---------------------------------------------------------------------------
# Holiday / market status
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def get_nse_holidays(target_year=None):
    session = init_nse_session()
    if session is None:
        return set()
    if target_year is None:
        target_year = get_ist_now().year
    holidays: set = set()
    try:
        payload = nse_get_json(session, NSE_HOLIDAY_URL, timeout=8, retries=3, backoff_seconds=0.8)
        if not payload:
            return holidays
        records = payload.get("FO", []) + payload.get("CM", [])
        for item in records:
            date_text = item.get("tradingDate") or item.get("date")
            if not date_text:
                continue
            try:
                holiday_date = datetime.datetime.strptime(date_text, "%d-%b-%Y").date()
                if holiday_date.year == target_year:
                    holidays.add(holiday_date)
            except ValueError:
                continue
    finally:
        session.close()
    return holidays


def is_nse_trading_holiday(check_date=None) -> bool:
    if check_date is None:
        check_date = get_ist_now().date()
    return check_date in get_nse_holidays(target_year=check_date.year)


def get_market_status(now=None):
    if now is None:
        now = get_ist_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=IST_ZONE)
    else:
        now = now.astimezone(IST_ZONE)
    if now.weekday() >= 5:
        return False, "Market is closed (weekend).", now
    if is_nse_trading_holiday(now.date()):
        return False, f"Market is closed (NSE holiday on {now.date().isoformat()}).", now
    market_open = datetime.time(9, 15)
    market_close = datetime.time(15, 30)
    if not (market_open <= now.time() <= market_close):
        return False, "Market is outside trading hours (09:15–15:30 IST).", now
    return True, "Market is open.", now


# ---------------------------------------------------------------------------
# Data freshness check
# ---------------------------------------------------------------------------
def is_data_fresh(data: pd.DataFrame, max_age_minutes: int = LIVE_DATA_MAX_CANDLE_AGE_MINUTES) -> bool:
    if data is None or data.empty:
        return False
    try:
        last_ts = data.index[-1]
        if getattr(last_ts, "tzinfo", None) is None:
            return True
        last_ts_ist = last_ts.astimezone(IST_ZONE)
        age_minutes = (get_ist_now() - last_ts_ist).total_seconds() / 60.0
        return age_minutes <= max_age_minutes
    except Exception:
        return True


# ---------------------------------------------------------------------------
# OPTIMIZATION 1: Kite-first data fetch with yfinance fallback
# ---------------------------------------------------------------------------
def fetch_intraday_data(ticker: str) -> pd.DataFrame:
    """
    Primary path  → Kite Connect 1-minute bars (real-time, no lag).
    Fallback path → yfinance with period/interval cascade.
    """
    # --- Primary: Kite Connect ---
    kite_data = fetch_intraday_data_kite(ticker)
    if not kite_data.empty and len(kite_data) >= 10:
        return kite_data

    # --- Fallback: yfinance ---
    last_error = None
    for _ in range(DATA_FETCH_RETRIES):
        for period, interval in (("1d", "1m"), ("2d", "1m"), ("5d", "5m")):
            try:
                data = yf.download(
                    tickers=ticker,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                )
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data = data.dropna(subset=["Close"]) if "Close" in data.columns else data
                if not data.empty:
                    return data
            except Exception as exc:
                last_error = exc
        try:
            data = yf.Ticker(ticker).history(period="1d", interval="1m", auto_adjust=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.dropna(subset=["Close"]) if "Close" in data.columns else data
            if not data.empty:
                return data
        except Exception as exc:
            last_error = exc
        time.sleep(DATA_FETCH_RETRY_SECONDS)

    if last_error:
        print(f"Live data fetch failed for {ticker}: {last_error}")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# OPTIMIZATION 7 & 8: Supertrend helper
# ---------------------------------------------------------------------------
def add_supertrend(data: pd.DataFrame, period: int = 7, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Compute Supertrend using the ATR already present in `data['ATR']`.
    Adds columns:
        Supertrend_Bull  bool  – price is above Supertrend line (uptrend)
        Supertrend_Bear  bool  – price is below Supertrend line (downtrend)
    """
    if "ATR" not in data.columns or data.empty:
        data["Supertrend_Bull"] = False
        data["Supertrend_Bear"] = False
        return data

    hl2 = (data["High"] + data["Low"]) / 2
    atr = data["ATR"]
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    direction = pd.Series(1, index=data.index)     # 1 = bull, -1 = bear

    closes = data["Close"].values
    fu = final_upper.values.copy()
    fl = final_lower.values.copy()
    bu = basic_upper.values
    bl = basic_lower.values
    dir_ = direction.values.copy()

    for i in range(1, len(data)):
        fu[i] = bu[i] if (bu[i] < fu[i - 1] or closes[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = bl[i] if (bl[i] > fl[i - 1] or closes[i - 1] < fl[i - 1]) else fl[i - 1]
        if closes[i] > fu[i - 1]:
            dir_[i] = 1
        elif closes[i] < fl[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]

    data["Supertrend_Bull"] = dir_ == 1
    data["Supertrend_Bear"] = dir_ == -1
    return data


# ---------------------------------------------------------------------------
# Core indicator calculation
# OPTIMIZATIONS: RSI(7), MACD(6,13,5), EMA34, VWAP anchored today, Supertrend
# ---------------------------------------------------------------------------
def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if data.empty or not required_cols.issubset(set(data.columns)):
        return data

    data = data.copy()

    # ------------------------------------------------------------------ RSI
    # OPTIMIZATION 5: RSI(7) – faster response on 1-minute bars
    rsi = RSIIndicator(close=data["Close"], window=7)
    data["RSI"] = rsi.rsi()

    # ----------------------------------------------------------------- MACD
    # OPTIMIZATION 4: MACD(6, 13, 5) tuned for 1-minute intraday bars
    macd_ind = MACD(
        close=data["Close"],
        window_slow=13,
        window_fast=6,
        window_sign=5,
    )
    data["MACD"] = macd_ind.macd()
    data["MACD_signal"] = macd_ind.macd_signal()
    data["MACD_hist"] = macd_ind.macd_diff()

    # --------------------------------------------------------------- Trends
    # OPTIMIZATION 6: EMA34 replaces EMA50 – initialises in 34 bars vs 50
    data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["EMA34"] = data["Close"].ewm(span=34, adjust=False).mean()

    # --------------------------------------------------------------- ADX
    adx = ADXIndicator(high=data["High"], low=data["Low"], close=data["Close"], window=14)
    data["ADX"] = adx.adx()

    # ----------------------------------------------------------- Volatility
    bb = BollingerBands(close=data["Close"], window=20, window_dev=2)
    data["BB_high"] = bb.bollinger_hband()
    data["BB_low"] = bb.bollinger_lband()

    atr_ind = AverageTrueRange(high=data["High"], low=data["Low"], close=data["Close"], window=14)
    data["ATR"] = atr_ind.average_true_range()
    data["ATR_pct"] = (data["ATR"] / data["Close"]) * 100

    # --------------------------------------------------------------- Volume
    data["Volume_SMA20"] = data["Volume"].rolling(20, min_periods=20).mean()
    obv = OnBalanceVolumeIndicator(close=data["Close"], volume=data["Volume"])
    data["OBV"] = obv.on_balance_volume()

    # --------------------------------------------------------------- VWAP
    # OPTIMIZATION 2: anchor VWAP to today's session only so multi-day
    # yfinance fallback data does not contaminate the VWAP baseline.
    today = get_ist_now().date()
    data["VWAP"] = float("nan")
    try:
        idx = data.index
        if hasattr(idx, "tz") and idx.tz is not None:
            local_idx = idx.tz_convert(IST_ZONE)
        else:
            local_idx = idx
        today_mask = pd.Series(
            [ts.date() == today for ts in local_idx],
            index=data.index,
        )
        today_data = data[today_mask]
        if not today_data.empty:
            tp = (today_data["High"] + today_data["Low"] + today_data["Close"]) / 3
            cum_tv = (tp * today_data["Volume"]).cumsum()
            cum_vol = today_data["Volume"].cumsum().replace(0, float("nan"))
            data.loc[today_data.index, "VWAP"] = (cum_tv / cum_vol).values
        else:
            # Fallback: full-series VWAP (acceptable for backtesting mode)
            tp = (data["High"] + data["Low"] + data["Close"]) / 3
            cum_tv = (tp * data["Volume"]).cumsum()
            cum_vol = data["Volume"].cumsum().replace(0, float("nan"))
            data["VWAP"] = (cum_tv / cum_vol).astype(float)
    except Exception:
        tp = (data["High"] + data["Low"] + data["Close"]) / 3
        cum_tv = (tp * data["Volume"]).cumsum()
        cum_vol = data["Volume"].cumsum().replace(0, float("nan"))
        data["VWAP"] = (cum_tv / cum_vol).astype(float)

    # OPTIMIZATION 8: Supertrend(7, 3.0)
    data = add_supertrend(data, period=7, multiplier=3.0)

    return data


# ---------------------------------------------------------------------------
# Signal generation
# OPTIMIZATIONS: volume spike 1.5×, EMA34, Supertrend in scoring (max → 13)
# ---------------------------------------------------------------------------
def generate_signals(data: pd.DataFrame, enable_shorting: bool = ENABLE_SHORTING) -> pd.DataFrame:
    if data.empty:
        return data

    data = data.copy()
    data["Signal"] = "HOLD"

    # -------------------------------------------------------- MACD crossover
    macd_above_signal = data["MACD"] > data["MACD_signal"]
    macd_above_prev = macd_above_signal.shift(1).fillna(False).astype(bool)
    macd_bull_cross = macd_above_signal & (~macd_above_prev)
    macd_bear_cross = (~macd_above_signal) & macd_above_prev
    macd_hist_rising = data["MACD_hist"] > data["MACD_hist"].shift(1)
    macd_hist_falling = data["MACD_hist"] < data["MACD_hist"].shift(1)

    # ------------------------------------------------------------ RSI slope
    rsi_delta = data["RSI"].diff()
    rsi_rising = rsi_delta > 0
    rsi_falling = rsi_delta < 0

    # ------------------------------------------------------- Trend: EMA34
    # OPTIMIZATION 6: use EMA34 everywhere EMA50 was used
    bullish_trend = (data["EMA20"] > data["EMA34"]) & (data["Close"] > data["VWAP"])
    bearish_trend = (data["EMA20"] < data["EMA34"]) & (data["Close"] < data["VWAP"])

    # OPTIMIZATION 7: volume spike threshold raised from 1.10× to 1.50×
    volume_spike = data["Volume"] > (data["Volume_SMA20"] * 1.50)

    adx_ok = data["ADX"] > 18
    atr_ok = data["ATR_pct"].between(0.25, 4.50)
    obv_up = data["OBV"].diff() > 0
    obv_down = data["OBV"].diff() < 0

    near_bb_low = data["Close"] <= (data["BB_low"] * 1.01)
    near_bb_high = data["Close"] >= (data["BB_high"] * 0.99)

    trend_gap_pct = ((data["EMA20"] - data["EMA34"]).abs() / data["Close"]) * 100
    strong_up_trend = bullish_trend & (data["ADX"] > 19) & (trend_gap_pct > 0.10)
    strong_down_trend = bearish_trend & (data["ADX"] > 21) & (trend_gap_pct > 0.12)

    # OPTIMIZATION 8: Supertrend included in confluence scoring
    supertrend_bull = data.get("Supertrend_Bull", pd.Series(False, index=data.index))
    supertrend_bear = data.get("Supertrend_Bear", pd.Series(False, index=data.index))

    # ------------------------------------------ Confluence scores (max = 13)
    # +2 MACD cross, +1 each remaining 11 factors, Supertrend adds 2 more
    long_score = (
        (macd_bull_cross * 2).astype(int)
        + ((data["RSI"] < 45) & rsi_rising).astype(int)
        + bullish_trend.astype(int)
        + (data["MACD_hist"] > 0).astype(int)
        + macd_hist_rising.astype(int)
        + volume_spike.astype(int)
        + adx_ok.astype(int)
        + near_bb_low.astype(int)
        + obv_up.astype(int)
        + atr_ok.astype(int)
        + supertrend_bull.astype(int)         # NEW: +1
        + supertrend_bull.astype(int)         # NEW: +1 (weighted ×2 total)
    )
    short_score = (
        (macd_bear_cross * 2).astype(int)
        + ((data["RSI"] > 55) & rsi_falling).astype(int)
        + bearish_trend.astype(int)
        + (data["MACD_hist"] < 0).astype(int)
        + macd_hist_falling.astype(int)
        + volume_spike.astype(int)
        + adx_ok.astype(int)
        + near_bb_high.astype(int)
        + obv_down.astype(int)
        + atr_ok.astype(int)
        + supertrend_bear.astype(int)         # NEW: +1
        + supertrend_bear.astype(int)         # NEW: +1 (weighted ×2 total)
    )

    score_max = 13   # was 11; Supertrend adds 2
    data["Long_Confidence"] = ((long_score / score_max) * 100).clip(0, 100).round(1)
    data["Short_Confidence"] = ((short_score / score_max) * 100).clip(0, 100).round(1)

    # ------------------------------------------------------ Entry masks
    if enable_shorting:
        buy_mask = (
            macd_bull_cross
            & strong_up_trend
            & (long_score >= 7)
            & data["RSI"].between(42, 62)
            & (data["Long_Confidence"] >= MIN_ENTRY_CONFIDENCE)
        )
    else:
        buy_mask = (
            macd_bull_cross
            & bullish_trend
            & macd_hist_rising
            & (long_score >= 6)
            & data["RSI"].between(40, 62)
            & (data["Long_Confidence"] >= MIN_ENTRY_CONFIDENCE)
            & (adx_ok | volume_spike)
        )
    short_mask = (
        macd_bear_cross
        & strong_down_trend
        & (short_score >= 7)
        & data["RSI"].between(43, 67)
        & (data["Short_Confidence"] >= MIN_ENTRY_CONFIDENCE)
    )

    exit_long_mask = (
        macd_bear_cross
        & ((data["RSI"] > 60) | (data["Close"] < data["VWAP"]) | (long_score <= 3))
    )
    cover_mask = (
        macd_bull_cross
        & ((data["RSI"] < 40) | (data["Close"] > data["VWAP"]) | (short_score <= 4))
    )

    # ------------------------------------------------------ Stateful loop
    position = None
    entry_price = None
    entry_i = -1
    peak_price = None
    realized_pnl_pct = 0.0
    stop_new_entries = False
    cooldown_bars = 0
    min_hold_bars = 2
    post_exit_cooldown = 2
    win_cooldown_bars = 1
    loss_cooldown_bars = 3 if enable_shorting else 4
    long_stop_atr = 0.9 if enable_shorting else 1.6
    long_target_atr = 2.0 if enable_shorting else 2.4
    long_trail_atr = 1.2 if enable_shorting else 1.4
    short_stop_atr = 0.9
    short_target_atr = 1.8

    for i in range(len(data)):
        if cooldown_bars > 0:
            cooldown_bars -= 1

        ts = data.index[i]
        try:
            ts_ist = ts.tz_convert(IST_ZONE) if getattr(ts, "tzinfo", None) is not None else ts
            t = ts_ist.time()
        except Exception:
            t = None

        long_entry_time_ok = True
        short_entry_time_ok = True
        if t is not None:
            if enable_shorting:
                long_entry_time_ok = datetime.time(9, 25) <= t <= datetime.time(15, 0)
            else:
                long_entry_time_ok = datetime.time(9, 16) <= t <= datetime.time(15, 5)
            short_entry_time_ok = datetime.time(11, 45) <= t <= datetime.time(14, 50)
        close_cutoff = datetime.time(15, 20) if enable_shorting else datetime.time(15, 25)
        force_exit_near_close = t is not None and t >= close_cutoff
        avoid_chop_window = False
        if (not enable_shorting) and t is not None:
            in_open_chop = datetime.time(10, 25) <= t < datetime.time(11, 0)
            in_afternoon_chop = datetime.time(14, 10) <= t < datetime.time(14, 25)
            avoid_chop_window = in_open_chop or in_afternoon_chop

        close_price = float(data["Close"].iloc[i])
        atr_value = float(data["ATR"].iloc[i]) if pd.notna(data["ATR"].iloc[i]) else 0.0

        if position == "LONG":
            bars_held = i - entry_i
            peak_price = close_price if peak_price is None else max(peak_price, close_price)
            stop_hit = atr_value > 0 and close_price <= (entry_price - long_stop_atr * atr_value)
            target_hit = atr_value > 0 and close_price >= (entry_price + long_target_atr * atr_value)
            trail_stop_hit = (
                atr_value > 0
                and peak_price > entry_price
                and close_price <= (peak_price - long_trail_atr * atr_value)
            )
            gave_back_after_runup = (
                atr_value > 0
                and peak_price >= (entry_price + atr_value)
                and close_price <= entry_price
            )
            rule_exit = bool(exit_long_mask.iloc[i]) and bars_held >= min_hold_bars

            if stop_hit or target_hit or trail_stop_hit or gave_back_after_runup or rule_exit or force_exit_near_close:
                data.iloc[i, data.columns.get_loc("Signal")] = "EXIT_LONG"
                trade_pnl_pct = ((close_price - entry_price) / entry_price) * 100 if entry_price else 0.0
                realized_pnl_pct += trade_pnl_pct
                if (not enable_shorting) and realized_pnl_pct <= -0.25:
                    stop_new_entries = True
                position = None
                entry_price = None
                entry_i = -1
                peak_price = None
                cooldown_bars = loss_cooldown_bars if trade_pnl_pct < 0 else win_cooldown_bars
            continue

        if position == "SHORT":
            bars_held = i - entry_i
            stop_hit = atr_value > 0 and close_price >= (entry_price + short_stop_atr * atr_value)
            target_hit = atr_value > 0 and close_price <= (entry_price - short_target_atr * atr_value)
            rule_exit = bool(cover_mask.iloc[i]) and bars_held >= min_hold_bars

            if stop_hit or target_hit or rule_exit or force_exit_near_close:
                data.iloc[i, data.columns.get_loc("Signal")] = "COVER"
                trade_pnl_pct = ((entry_price - close_price) / entry_price) * 100 if entry_price else 0.0
                realized_pnl_pct += trade_pnl_pct
                position = None
                entry_price = None
                entry_i = -1
                peak_price = None
                cooldown_bars = loss_cooldown_bars if trade_pnl_pct < 0 else post_exit_cooldown
            continue

        if cooldown_bars > 0 or stop_new_entries:
            continue

        if bool(buy_mask.iloc[i]) and long_entry_time_ok and (not avoid_chop_window):
            data.iloc[i, data.columns.get_loc("Signal")] = "BUY"
            position = "LONG"
            entry_price = close_price
            entry_i = i
            peak_price = close_price
            continue

        if enable_shorting and bool(short_mask.iloc[i]) and short_entry_time_ok:
            data.iloc[i, data.columns.get_loc("Signal")] = "SHORT"
            position = "SHORT"
            entry_price = close_price
            entry_i = i
            peak_price = None
            continue

    if not enable_shorting:
        data.loc[data["Signal"] == "SHORT", "Signal"] = "SELL"
        data.loc[data["Signal"] == "COVER", "Signal"] = "BUY"
        data.loc[data["Signal"] == "EXIT_LONG", "Signal"] = "SELL"

    data["Signal_Confidence"] = 0.0
    data.loc[data["Signal"].isin(["BUY", "COVER"]), "Signal_Confidence"] = data["Long_Confidence"]
    data.loc[data["Signal"].isin(["SHORT", "EXIT_LONG", "SELL"]), "Signal_Confidence"] = data["Short_Confidence"]
    return data


# ---------------------------------------------------------------------------
# Breakout universe scanner
# OPTIMIZATION 9: ORB opening range uses time-based filtering
# ---------------------------------------------------------------------------
def _scan_breakout_candidate(stock: str):
    try:
        data = fetch_intraday_data(stock)
        if data.empty or len(data) < 60:
            return None
        if not {"Open", "High", "Low", "Close", "Volume"}.issubset(set(data.columns)):
            return None

        data = data.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
        if data.empty or len(data) < 60:
            return None

        ts_index = data.index
        if hasattr(ts_index, "tz") and ts_index.tz is not None:
            local_idx = ts_index.tz_convert(IST_ZONE)
        else:
            local_idx = ts_index
        data.index = local_idx
        if not is_data_fresh(data):
            return None

        intraday = data.between_time("09:15", "15:30").copy()
        if intraday.empty or len(intraday) < 45:
            return None

        intraday["EMA20"] = intraday["Close"].ewm(span=20, adjust=False).mean()
        intraday["Volume_SMA20"] = intraday["Volume"].rolling(20, min_periods=20).mean()
        intraday_volume = float(intraday["Volume"].sum())
        intraday_traded_value = float((intraday["Close"] * intraday["Volume"]).sum())
        if intraday_volume < MIN_DAILY_VOLUME or intraday_traded_value < MIN_DAILY_TRADED_VALUE:
            return None

        # OPTIMIZATION 9: time-based ORB instead of iloc[:N]
        # This is robust to gaps in yfinance 1-minute data.
        session_start = intraday.index[0]
        orb_end_time = (session_start + datetime.timedelta(minutes=OPENING_RANGE_MINUTES))
        open_range = intraday[intraday.index <= orb_end_time]
        if open_range.empty:
            open_range = intraday.iloc[:OPENING_RANGE_MINUTES]  # fallback

        or_high = float(open_range["High"].max())
        or_low = float(open_range["Low"].min())

        last_close = float(intraday["Close"].iloc[-1])
        if last_close < MIN_TRADABLE_PRICE:
            return None
        last_volume = float(intraday["Volume"].iloc[-1])
        last_vol_sma20 = (
            float(intraday["Volume_SMA20"].iloc[-1])
            if pd.notna(intraday["Volume_SMA20"].iloc[-1])
            else 0.0
        )
        ema20 = float(intraday["EMA20"].iloc[-1])
        recent_closes = (
            intraday["Close"].iloc[-BREAKOUT_CONFIRM_BARS:]
            if len(intraday) >= BREAKOUT_CONFIRM_BARS
            else intraday["Close"].iloc[-1:]
        )
        orb_level = or_high * 1.001
        orb_confirmed = bool((recent_closes > orb_level).all())

        orb_up = (
            last_close > orb_level
            and last_close > ema20
            and last_vol_sma20 > 0
            and last_volume > (last_vol_sma20 * 1.05)
            and orb_confirmed
        )

        cons_window = intraday.tail(CONSOLIDATION_LOOKBACK_BARS + 1)
        cons_breakout = False
        if len(cons_window) >= CONSOLIDATION_LOOKBACK_BARS + 1:
            pre_break = cons_window.iloc[:-1]
            range_pct = (
                (float(pre_break["High"].max()) - float(pre_break["Low"].min()))
                / max(last_close, 1e-6)
            ) * 100
            pre_high = float(pre_break["High"].max())
            cons_level = pre_high * 1.001
            cons_confirmed = bool((recent_closes > cons_level).all())
            cons_breakout = (
                range_pct <= CONSOLIDATION_RANGE_PCT_MAX
                and last_close > cons_level
                and last_close > ema20
                and cons_confirmed
            )

        if not (orb_up or cons_breakout):
            return None

        score = 0
        labels = []
        if orb_up:
            score += 2
            labels.append("ORB")
        if cons_breakout:
            score += 2
            labels.append("CONS")
        if last_close > ema20:
            score += 1
        if last_vol_sma20 > 0 and last_volume > last_vol_sma20:
            score += 1
        if intraday_traded_value >= (MIN_DAILY_TRADED_VALUE * 2):
            score += 1

        return {
            "stock": stock,
            "score": score,
            "labels": "+".join(labels),
            "last_close": last_close,
            "or_range_pct": ((or_high - or_low) / max(last_close, 1e-6)) * 100,
            "day_traded_value": intraday_traded_value,
        }
    except Exception:
        return None


def get_dynamic_stocks_with_meta(enable_shorting: bool = ENABLE_SHORTING):
    indexes = {
        "NIFTY50": [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS",
            "BAJFINANCE.NS", "BHARTIARTL.NS", "CIPLA.NS", "DRREDDY.NS", "GRASIM.NS",
            "HCLTECH.NS", "HEROMOTOCO.NS", "INDUSINDBK.NS", "JSWSTEEL.NS", "SBIN.NS",
        ],
        "NIFTY NEXT 50": [
            "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "ADANITRANS.NS",
            "AMBUJACEM.NS", "APOLLOHOSP.NS", "AUROPHARMA.NS", "BAJAJHLDNG.NS",
            "BANDHANBNK.NS", "BERGEPAINT.NS", "BIOCON.NS", "BOSCHLTD.NS",
            "CHOLAFIN.NS", "COLPAL.NS", "DABUR.NS", "DLF.NS", "DMART.NS",
            "GLAND.NS", "GODREJPROP.NS", "HAVELLS.NS",
        ],
        "MIDCAP": [
            "GODREJCP.NS", "VOLTAS.NS", "PAGEIND.NS", "BAJAJ-AUTO.NS", "MARUTI.NS",
            "EICHERMOT.NS", "M&M.NS", "ULTRACEMCO.NS", "BAJAJFINSV.NS", "DIVISLAB.NS",
            "SHREECEM.NS", "TITAN.NS", "UPL.NS", "MCDOWELL-N.NS", "PIDILITIND.NS",
        ],
        "SMALLCAP": [
            "COALINDIA.NS", "NMDC.NS", "SAIL.NS", "BEL.NS", "NTPC.NS",
            "GAIL.NS", "ONGC.NS", "POWERGRID.NS", "IOC.NS", "BPCL.NS",
            "SUNPHARMA.NS", "TATACONSUM.NS", "BRITANNIA.NS", "NESTLEIND.NS", "ADANIPORTS.NS",
        ],
    }
    all_tickers = list(dict.fromkeys(stock for group in indexes.values() for stock in group))

    breakout_candidates = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_scan_breakout_candidate, stock): stock for stock in all_tickers}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                breakout_candidates.append(result)

    if breakout_candidates:
        breakout_candidates.sort(key=lambda x: (x["score"], x["last_close"]), reverse=True)
        selected = [item["stock"] for item in breakout_candidates[:BREAKOUT_UNIVERSE_SIZE]]
        labels = ", ".join(
            f"{item['stock']}[{item['labels']}]" for item in breakout_candidates[:5]
        )
        return selected, "breakout_scan", f"Selected via ORB/consolidation breakout scan. Top: {labels}"

    source = "nse_momentum_fallback"
    all_changes = get_dynamic_stocks_from_nse(all_tickers)
    if not all_changes:
        source = "yfinance_momentum_fallback"
        print("Breakout scan found no candidates. Falling back to momentum selector via yfinance.")
        all_changes = get_dynamic_stocks_from_yfinance(all_tickers)
    if not all_changes:
        return FALLBACK_STOCKS.copy(), "fixed_fallback", "Breakout scan and momentum selector unavailable."

    all_changes = [(stock, float(change)) for stock, change in all_changes if pd.notna(change)]
    if not all_changes:
        return FALLBACK_STOCKS.copy(), "fixed_fallback", "No numeric values from fallback selector."

    all_changes.sort(key=lambda x: x[1], reverse=True)
    top_gainers = [stock for stock, _ in all_changes[:LONG_ONLY_TOP_GAINERS]]
    all_changes.sort(key=lambda x: x[1])
    top_losers = [stock for stock, _ in all_changes[:5]]
    selected = top_gainers + top_losers if enable_shorting else top_gainers
    return list(dict.fromkeys(selected)), source, "Fallback selector used."


def get_dynamic_stocks(enable_shorting: bool = ENABLE_SHORTING):
    stocks, _, _ = get_dynamic_stocks_with_meta(enable_shorting=enable_shorting)
    return stocks


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    market_is_open, market_message, _ = get_market_status()
    if not market_is_open:
        print(market_message)
        return

    stocks = get_dynamic_stocks(enable_shorting=ENABLE_SHORTING)
    print(f"Selected stocks for analysis: {stocks}")
    print(f"Round-trip transaction cost model: {ROUND_TRIP_COST_PCT*100:.4f}% of capital used")

    for stock in stocks:
        try:
            data = fetch_intraday_data(stock)
            if data.empty or len(data) < 35:
                print(f"No data for {stock}")
                continue

            data = calculate_indicators(data)
            data = generate_signals(data)

            # EMA34 replaces EMA50 in dropna guard
            data = data.dropna(
                subset=["RSI", "MACD", "MACD_signal", "Close", "EMA20", "EMA34", "VWAP", "ADX"]
            )
            if data.empty:
                print(f"Insufficient indicator data for {stock}")
                continue

            latest_signal = data["Signal"].iloc[-1]
            latest_price = float(data["Close"].iloc[-1])
            latest_confidence = float(data["Signal_Confidence"].iloc[-1])
            st_bull = bool(data["Supertrend_Bull"].iloc[-1])

            print(
                f"{stock}: Signal={latest_signal}  Conf={latest_confidence:.1f}%  "
                f"Price={latest_price:.2f}  Supertrend={'BULL' if st_bull else 'BEAR'}"
            )

        except Exception as e:
            print(f"Error processing {stock}: {e}")


if __name__ == "__main__":
    main()
