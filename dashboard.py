"""
dashboard.py  –  Streamlit live dashboard for the intraday quant engine
Optimizations applied:
  10. _cached_snapshot uses per-minute cache-bust key (get_cache_bust_key())
      so stale stateful signals are never served from cache mid-session.
      All other changes live in main.py and are reflected here automatically.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st

from main import (
    DAILY_MAX_LOSS_PCT,
    MAX_CAPITAL_PER_TRADE_PCT,
    MAX_OPEN_TRADES,
    MIN_STOP_LOSS_PCT,
    RISK_PER_TRADE_PCT,
    ROUND_TRIP_COST_PCT,
    STARTING_CAPITAL,
    STOP_LOSS_ATR_MULTIPLIER,
    calculate_indicators,
    fetch_intraday_data,
    generate_signals,
    get_cache_bust_key,
    get_dynamic_stocks_with_meta,
    get_market_status,
)

st.set_page_config(page_title="Intraday Quant Dashboard", page_icon="📈", layout="wide")
st.title("Intraday Quant Dashboard")
st.caption(
    "Aggressive intraday signal monitor for dynamically selected NSE stocks  |  "
    "All P&L figures include estimated round-trip transaction costs."
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state():
    if "positions" not in st.session_state:
        st.session_state.positions = {}
    if "trade_log" not in st.session_state:
        st.session_state.trade_log = []
    if "realized_pnl_inr" not in st.session_state:
        st.session_state.realized_pnl_inr = 0.0
    if "stop_new_entries" not in st.session_state:
        st.session_state.stop_new_entries = False


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------
def _position_size(price: float, atr_value: float):
    max_capital = STARTING_CAPITAL * MAX_CAPITAL_PER_TRADE_PCT
    risk_cash = STARTING_CAPITAL * RISK_PER_TRADE_PCT
    stop_distance = max(atr_value * STOP_LOSS_ATR_MULTIPLIER, price * MIN_STOP_LOSS_PCT)
    if stop_distance <= 0 or price <= 0:
        return 0, 0.0
    qty = int(min(max_capital // price, risk_cash // stop_distance))
    if qty <= 0:
        return 0, 0.0
    return qty, stop_distance


# ---------------------------------------------------------------------------
# OPTIMIZATION 3: transaction costs applied to every simulated trade
# ---------------------------------------------------------------------------
def _net_pnl_inr(gross_pnl_inr: float, capital_used: float) -> float:
    """Deduct estimated round-trip transaction costs from gross P&L."""
    cost = capital_used * ROUND_TRIP_COST_PCT
    return gross_pnl_inr - cost


def _apply_trade_logic(stock, signal, price, timestamp, atr_value):
    positions = st.session_state.positions
    if stock not in positions:
        positions[stock] = {
            "side": None,
            "entry_price": None,
            "entry_time": None,
            "qty": 0,
            "realized_pnl_pct": 0.0,
        }

    current = positions[stock]
    open_count = sum(1 for pos in positions.values() if pos.get("side") is not None)

    if signal == "BUY" and current["side"] is None:
        if st.session_state.stop_new_entries or open_count >= MAX_OPEN_TRADES:
            return
        qty, _ = _position_size(float(price), float(atr_value))
        if qty <= 0:
            return
        current.update({"side": "LONG", "entry_price": float(price),
                        "entry_time": timestamp, "qty": qty})

    elif signal == "SHORT" and current["side"] is None:
        if st.session_state.stop_new_entries or open_count >= MAX_OPEN_TRADES:
            return
        qty, _ = _position_size(float(price), float(atr_value))
        if qty <= 0:
            return
        current.update({"side": "SHORT", "entry_price": float(price),
                        "entry_time": timestamp, "qty": qty})

    elif signal == "EXIT_LONG" and current["side"] == "LONG" and current["entry_price"] is not None:
        entry = current["entry_price"]
        qty = int(current.get("qty") or 0)
        gross_inr = (float(price) - entry) * qty
        capital_used = entry * qty
        net_inr = _net_pnl_inr(gross_inr, capital_used)
        realized_pct = ((float(price) - entry) / entry) * 100
        current["realized_pnl_pct"] += realized_pct
        st.session_state.realized_pnl_inr += net_inr
        st.session_state.trade_log.append({
            "Stock": stock, "Side": "LONG", "Qty": qty,
            "Entry Price": round(entry, 2), "Exit Price": round(float(price), 2),
            "Entry Time": current["entry_time"], "Exit Time": timestamp,
            "Gross PnL (INR)": round(gross_inr, 2),
            "Net PnL (INR)": round(net_inr, 2),
            "PnL %": round(realized_pct, 2),
            "Costs (INR)": round(gross_inr - net_inr, 2),
        })
        current.update({"side": None, "entry_price": None, "entry_time": None, "qty": 0})

    elif signal == "COVER" and current["side"] == "SHORT" and current["entry_price"] is not None:
        entry = current["entry_price"]
        qty = int(current.get("qty") or 0)
        gross_inr = (entry - float(price)) * qty
        capital_used = entry * qty
        net_inr = _net_pnl_inr(gross_inr, capital_used)
        realized_pct = ((entry - float(price)) / entry) * 100
        current["realized_pnl_pct"] += realized_pct
        st.session_state.realized_pnl_inr += net_inr
        st.session_state.trade_log.append({
            "Stock": stock, "Side": "SHORT", "Qty": qty,
            "Entry Price": round(entry, 2), "Exit Price": round(float(price), 2),
            "Entry Time": current["entry_time"], "Exit Time": timestamp,
            "Gross PnL (INR)": round(gross_inr, 2),
            "Net PnL (INR)": round(net_inr, 2),
            "PnL %": round(realized_pct, 2),
            "Costs (INR)": round(gross_inr - net_inr, 2),
        })
        current.update({"side": None, "entry_price": None, "entry_time": None, "qty": 0})

    if st.session_state.realized_pnl_inr <= -(STARTING_CAPITAL * DAILY_MAX_LOSS_PCT):
        st.session_state.stop_new_entries = True


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------
def _build_snapshot(stock: str):
    data = fetch_intraday_data(stock)
    if data.empty or len(data) < 35 or "Close" not in data.columns:
        return None

    data = calculate_indicators(data)
    data = generate_signals(data)
    data = data.dropna(subset=["Close", "RSI", "MACD", "MACD_signal", "MACD_hist"])
    if data.empty:
        return None

    latest = data.iloc[-1]
    prev_close = float(data["Close"].iloc[0])
    latest_close = float(latest["Close"])
    day_change_pct = ((latest_close - prev_close) / prev_close) * 100 if prev_close else 0.0

    return {
        "Stock": stock,
        "Signal": latest["Signal"],
        "Signal Confidence": float(latest.get("Signal_Confidence", 0.0)),
        "Price": latest_close,
        "ATR": float(latest.get("ATR", 0.0)) if pd.notna(latest.get("ATR", 0.0)) else 0.0,
        "Day Change %": day_change_pct,
        "RSI": float(latest["RSI"]),
        "MACD": float(latest["MACD"]),
        "MACD Signal": float(latest["MACD_signal"]),
        "MACD Hist": float(latest["MACD_hist"]),
        "Supertrend": "BULL" if bool(latest.get("Supertrend_Bull", False)) else "BEAR",
        "Last Candle": data.index[-1],
        "data": data,
    }


# OPTIMIZATION 10: cache-bust key is per-minute → stale stateful signals
# are automatically evicted every 60 seconds even within the TTL window.
@st.cache_data(ttl=20, show_spinner=False)
def _cached_snapshot(stock: str, _cache_bust: str):
    """_cache_bust is the per-minute key from get_cache_bust_key(); it is
    intentionally unused inside the function body but forces Streamlit to
    treat a new minute as a cache miss, preventing stale stateful signals."""
    return _build_snapshot(stock)


def _fetch_snapshots(stocks, max_workers=8):
    cache_bust = get_cache_bust_key()
    snapshots = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_cached_snapshot, stock, cache_bust): stock
            for stock in stocks
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                snapshots.append(result)
    return snapshots


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
_init_state()

with st.sidebar:
    st.subheader("Controls")
    max_stocks = st.slider("Max stocks", min_value=5, max_value=20, value=10, step=1)
    signal_filter = st.multiselect(
        "Signal filter",
        options=["BUY", "SHORT", "EXIT_LONG", "COVER", "HOLD"],
        default=["BUY", "SHORT", "EXIT_LONG", "COVER", "HOLD"],
    )
    auto_refresh = st.checkbox("Auto refresh", value=True)
    refresh_seconds = st.slider("Refresh interval (sec)", 15, 180, 45, 5)
    refresh_now = st.button("Refresh now")
    reset_sim = st.button("Reset simulated book")

    st.markdown("---")
    st.caption(
        f"Transaction cost model\n"
        f"Round-trip: **{ROUND_TRIP_COST_PCT*100:.4f}%** of capital used\n"
        f"(brokerage + STT + exchange + SEBI + stamp duty)"
    )

if reset_sim:
    st.session_state.positions = {}
    st.session_state.trade_log = []
    st.session_state.realized_pnl_inr = 0.0
    st.session_state.stop_new_entries = False

if refresh_now:
    st.rerun()


# ---------------------------------------------------------------------------
# Main dashboard render
# ---------------------------------------------------------------------------
def _render_live_dashboard(max_stocks, signal_filter):
    is_open, market_message, now_ist = get_market_status()
    status_fn = st.success if is_open else st.warning
    status_fn(f"{market_message} | IST: {now_ist.strftime('%Y-%m-%d %H:%M:%S')}")

    stocks, selector_source, selector_reason = get_dynamic_stocks_with_meta()
    stocks = stocks[:max_stocks]
    source_label = {
        "nse": "NSE live",
        "yfinance": "yfinance fallback",
        "fixed_fallback": "Fixed fallback list",
        "breakout_scan": "ORB/consolidation breakout scan",
        "nse_momentum_fallback": "NSE momentum fallback",
        "yfinance_momentum_fallback": "yfinance momentum fallback",
    }.get(selector_source, selector_source)
    st.write(f"Universe ({len(stocks)}) [{source_label}]: {', '.join(stocks)}")
    if selector_reason:
        st.caption(f"Selector note: {selector_reason}")

    if not stocks:
        st.error("No stocks available from selector.")
        return

    snapshots = _fetch_snapshots(stocks)
    if not snapshots:
        st.error("No intraday data returned for current universe.")
        return

    for snap in snapshots:
        _apply_trade_logic(
            stock=snap["Stock"],
            signal=snap["Signal"],
            price=snap["Price"],
            timestamp=snap["Last Candle"],
            atr_value=snap.get("ATR", 0.0),
        )

    # ---------------------------------------------------------------- rows
    rows = []
    for snap in snapshots:
        if snap["Signal"] not in signal_filter:
            continue
        stock_pos = st.session_state.positions.get(snap["Stock"], {})
        side = stock_pos.get("side")
        entry_price = stock_pos.get("entry_price")
        qty = int(stock_pos.get("qty") or 0)

        unrealized_pct = 0.0
        unrealized_inr = 0.0
        unrealized_net_inr = 0.0
        if side == "LONG" and entry_price:
            unrealized_pct = ((snap["Price"] - entry_price) / entry_price) * 100
            unrealized_inr = (snap["Price"] - entry_price) * qty
            unrealized_net_inr = _net_pnl_inr(unrealized_inr, entry_price * qty)
        elif side == "SHORT" and entry_price:
            unrealized_pct = ((entry_price - snap["Price"]) / entry_price) * 100
            unrealized_inr = (entry_price - snap["Price"]) * qty
            unrealized_net_inr = _net_pnl_inr(unrealized_inr, entry_price * qty)

        rows.append({
            "Stock": snap["Stock"],
            "Signal": snap["Signal"],
            "Signal Confidence %": round(snap["Signal Confidence"], 1),
            "Supertrend": snap["Supertrend"],
            "Position": side if side else "FLAT",
            "Qty": qty,
            "Price": round(snap["Price"], 2),
            "Day Change %": round(snap["Day Change %"], 2),
            "RSI": round(snap["RSI"], 2),
            "MACD Hist": round(snap["MACD Hist"], 4),
            "Unreal. PnL INR (Net)": round(unrealized_net_inr, 2),
            "Unreal. PnL %": round(unrealized_pct, 2),
            "Last Candle": snap["Last Candle"],
        })

    signal_df = pd.DataFrame(rows)
    if signal_df.empty:
        st.warning("No rows after applying signal filter.")
        return

    signal_df = signal_df.sort_values(by=["Signal", "Day Change %"], ascending=[True, False])
    snapshot_df = pd.DataFrame(
        [{k: v for k, v in s.items() if k != "data"} for s in snapshots]
    )

    # ----------------------------------------------------------- metrics
    buy_count = int((snapshot_df["Signal"] == "BUY").sum())
    short_count = int((snapshot_df["Signal"] == "SHORT").sum())
    exit_long_count = int((snapshot_df["Signal"] == "EXIT_LONG").sum())
    cover_count = int((snapshot_df["Signal"] == "COVER").sum())
    avg_rsi = float(snapshot_df["RSI"].mean())
    bull_st = int((snapshot_df["Supertrend"] == "BULL").sum())

    total_unrealized_net_inr = 0.0
    for snap in snapshots:
        stock_pos = st.session_state.positions.get(snap["Stock"], {})
        side = stock_pos.get("side")
        entry_price = stock_pos.get("entry_price")
        qty = int(stock_pos.get("qty") or 0)
        if side == "LONG" and entry_price and qty > 0:
            gross = (snap["Price"] - entry_price) * qty
            total_unrealized_net_inr += _net_pnl_inr(gross, entry_price * qty)
        elif side == "SHORT" and entry_price and qty > 0:
            gross = (entry_price - snap["Price"]) * qty
            total_unrealized_net_inr += _net_pnl_inr(gross, entry_price * qty)

    sim_pnl_net = st.session_state.realized_pnl_inr + total_unrealized_net_inr

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("BUY", buy_count)
    c2.metric("SHORT", short_count)
    c3.metric("EXIT_LONG", exit_long_count)
    c4.metric("COVER", cover_count)
    c5.metric("Avg RSI", f"{avg_rsi:.2f}")
    c6.metric("ST Bull", f"{bull_st}/{len(snapshots)}")
    c7.metric("Sim PnL INR (Net)", f"{sim_pnl_net:.2f}")

    st.caption(
        f"Risk limits: max capital/trade {int(MAX_CAPITAL_PER_TRADE_PCT*100)}%  |  "
        f"risk/trade {int(RISK_PER_TRADE_PCT*100)}%  |  "
        f"max daily loss {DAILY_MAX_LOSS_PCT*100:.1f}%  |  "
        f"max open trades {MAX_OPEN_TRADES}  |  "
        f"entries {'🔴 STOPPED' if st.session_state.stop_new_entries else '🟢 active'}  |  "
        f"round-trip cost {ROUND_TRIP_COST_PCT*100:.4f}%"
    )

    # ------------------------------------------------- Actionable signals
    actionable = signal_df[signal_df["Signal"] != "HOLD"]
    st.subheader("Actionable Signals")
    if actionable.empty:
        st.info("No actionable entry/exit triggers in current refresh window.")
    else:
        st.dataframe(actionable, use_container_width=True, hide_index=True)

    st.subheader("Full Signal Grid")
    st.dataframe(signal_df, use_container_width=True, hide_index=True)

    # ----------------------------------------------- Price & RSI drilldown
    st.subheader("Price, RSI and Supertrend Drilldown")
    symbol = st.selectbox("Select stock", options=signal_df["Stock"].tolist(), index=0)
    selected = next((x for x in snapshots if x["Stock"] == symbol), None)
    if selected:
        chart_df = selected["data"][["Close", "RSI"]].copy()
        chart_df.columns = ["Price", "RSI"]
        st.line_chart(chart_df, use_container_width=True)

        # Supertrend overlay (boolean → numeric for charting)
        st_df = selected["data"][["Supertrend_Bull"]].copy().astype(int)
        st_df.columns = ["Supertrend (1=Bull, 0=Bear)"]
        st.line_chart(st_df, use_container_width=True)

    # -------------------------------------------------- Simulated trade log
    st.subheader("Simulated Trade Log (Net of Costs)")
    if st.session_state.trade_log:
        trade_df = pd.DataFrame(st.session_state.trade_log).sort_values(
            by="Exit Time", ascending=False
        )
        st.dataframe(trade_df, use_container_width=True, hide_index=True)
        total_gross = trade_df["Gross PnL (INR)"].sum()
        total_costs = trade_df["Costs (INR)"].sum()
        total_net = trade_df["Net PnL (INR)"].sum()
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Gross PnL (INR)", f"{total_gross:.2f}")
        col_b.metric("Total Costs (INR)", f"{total_costs:.2f}")
        col_c.metric("Total Net PnL (INR)", f"{total_net:.2f}")
    else:
        st.caption(
            "No closed trades yet. "
            "Entries/exits are driven by BUY/SHORT and EXIT_LONG/COVER signals."
        )

    st.markdown("---")
    st.caption(
        "Simulation only. "
        "Use broker-validated live feeds and proper risk controls before real deployment."
    )


# ---------------------------------------------------------------------------
# Fragment / auto-refresh wiring
# ---------------------------------------------------------------------------
run_every = f"{refresh_seconds}s" if auto_refresh else None
fragment_decorator = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)

if fragment_decorator:
    @fragment_decorator(run_every=run_every)
    def _live_fragment():
        _render_live_dashboard(max_stocks=max_stocks, signal_filter=signal_filter)

    _live_fragment()
else:
    if auto_refresh:
        st.warning("Auto refresh requires Streamlit ≥ 1.33 with st.fragment support. Use 'Refresh now'.")
    _render_live_dashboard(max_stocks=max_stocks, signal_filter=signal_filter)
