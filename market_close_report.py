"""
market_close_report.py  –  Post-market EOD report generator
Optimizations applied:
  3.  Transaction cost model: every trade P&L is reported both gross and net.
      Net figure uses ROUND_TRIP_COST_PCT imported from main.py.
  All other signal/indicator improvements are pulled in via main.py imports.
"""

import argparse
import datetime
from pathlib import Path

import pandas as pd

from main import (
    DAILY_MAX_LOSS_PCT,
    IST_ZONE,
    MAX_CAPITAL_PER_TRADE_PCT,
    MAX_OPEN_TRADES,
    MIN_STOP_LOSS_PCT,
    RISK_PER_TRADE_PCT,
    ROUND_TRIP_COST_PCT,
    STARTING_CAPITAL,
    STOP_LOSS_ATR_MULTIPLIER,
    ENABLE_SHORTING,
    calculate_indicators,
    fetch_intraday_data,
    generate_signals,
    get_dynamic_stocks,
    get_ist_now,
    is_nse_trading_holiday,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_after_market_close(now_ist: datetime.datetime) -> bool:
    return now_ist.time() >= datetime.time(15, 30)


def is_trading_day(now_ist: datetime.datetime) -> bool:
    if now_ist.weekday() >= 5:
        return False
    if is_nse_trading_holiday(now_ist.date()):
        return False
    return True


def signal_side(signal: str) -> str:
    if signal in ("BUY", "COVER"):
        return "BUY"
    if signal in ("SHORT", "EXIT_LONG", "SELL"):
        return "SELL"
    return "HOLD"


def build_signal_events(stock: str, data: pd.DataFrame) -> list[dict]:
    rows = []
    for ts, row in data.iterrows():
        signal = row["Signal"]
        if signal == "HOLD":
            continue
        price = float(row["Close"])
        confidence = float(row.get("Signal_Confidence", 0.0))
        atr = float(row.get("ATR", 0.0)) if pd.notna(row.get("ATR", 0.0)) else 0.0
        supertrend = "BULL" if bool(row.get("Supertrend_Bull", False)) else "BEAR"
        rows.append({
            "Stock": stock,
            "Timestamp": ts,
            "Signal": signal,
            "Signal Side": signal_side(signal),
            "Price": price,
            "Confidence %": confidence,
            "ATR": atr,
            "Supertrend": supertrend,
        })
    return rows


def calculate_position_size(
    price: float, atr_value: float
) -> tuple[int, float, float]:
    max_capital = STARTING_CAPITAL * MAX_CAPITAL_PER_TRADE_PCT
    risk_cash = STARTING_CAPITAL * RISK_PER_TRADE_PCT
    stop_distance = max(atr_value * STOP_LOSS_ATR_MULTIPLIER, price * MIN_STOP_LOSS_PCT)
    if stop_distance <= 0:
        return 0, 0.0, 0.0
    qty_by_capital = int(max_capital // price) if price > 0 else 0
    qty_by_risk = int(risk_cash // stop_distance)
    qty = min(qty_by_capital, qty_by_risk)
    if qty <= 0:
        return 0, 0.0, stop_distance
    capital_used = qty * price
    max_risk_inr = qty * stop_distance
    return qty, capital_used, max_risk_inr


# ---------------------------------------------------------------------------
# OPTIMIZATION 3: net P&L helper
# ---------------------------------------------------------------------------
def _net_pnl(gross_pnl_inr: float, capital_used: float) -> float:
    """Deduct estimated round-trip transaction costs."""
    return gross_pnl_inr - capital_used * ROUND_TRIP_COST_PCT


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------
def simulate_portfolio_day(
    signal_events: list[dict], stock_data: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_events = sorted(signal_events, key=lambda x: (x["Timestamp"], x["Stock"]))
    executed_signal_events: list[dict] = []
    trades: list[dict] = []

    open_positions: dict[str, dict] = {}
    realized_pnl_inr = 0.0
    stop_new_entries = False

    for event in signal_events:
        stock = event["Stock"]
        ts = event["Timestamp"]
        signal = event["Signal"]
        price = float(event["Price"])
        atr = float(event.get("ATR", 0.0))
        side = event["Signal Side"]
        confidence = float(event.get("Confidence %", 0.0))
        supertrend = event.get("Supertrend", "N/A")

        executed = "IGNORED"
        reason = ""

        if signal in ("EXIT_LONG", "SELL"):
            if stock in open_positions and open_positions[stock]["side"] == "LONG":
                pos = open_positions.pop(stock)
                gross_pnl = (price - pos["entry_price"]) * pos["Qty"]
                net_pnl = _net_pnl(gross_pnl, pos["capital_used"])
                realized_pnl_inr += net_pnl
                trades.append({
                    "Stock": stock,
                    "Side": "LONG",
                    "Entry Time": pos["entry_time"],
                    "Exit Time": ts,
                    "Qty": pos["Qty"],
                    "Entry Price": round(pos["entry_price"], 2),
                    "Exit Price": round(price, 2),
                    "Capital Used": round(pos["capital_used"], 2),
                    "Risk at Entry (INR)": round(pos["risk_inr"], 2),
                    "Exit Signal": signal,
                    "Gross PnL (INR)": round(gross_pnl, 2),
                    "Costs (INR)": round(gross_pnl - net_pnl, 2),
                    "Net PnL (INR)": round(net_pnl, 2),
                    "PnL %": round((gross_pnl / max(pos["capital_used"], 1e-6)) * 100, 2),
                    "Net PnL %": round((net_pnl / max(pos["capital_used"], 1e-6)) * 100, 2),
                    "Status": "CLOSED",
                })
                executed = "EXITED"
            else:
                reason = "No open LONG"

        elif signal == "COVER":
            if stock in open_positions and open_positions[stock]["side"] == "SHORT":
                pos = open_positions.pop(stock)
                gross_pnl = (pos["entry_price"] - price) * pos["Qty"]
                net_pnl = _net_pnl(gross_pnl, pos["capital_used"])
                realized_pnl_inr += net_pnl
                trades.append({
                    "Stock": stock,
                    "Side": "SHORT",
                    "Entry Time": pos["entry_time"],
                    "Exit Time": ts,
                    "Qty": pos["Qty"],
                    "Entry Price": round(pos["entry_price"], 2),
                    "Exit Price": round(price, 2),
                    "Capital Used": round(pos["capital_used"], 2),
                    "Risk at Entry (INR)": round(pos["risk_inr"], 2),
                    "Exit Signal": signal,
                    "Gross PnL (INR)": round(gross_pnl, 2),
                    "Costs (INR)": round(gross_pnl - net_pnl, 2),
                    "Net PnL (INR)": round(net_pnl, 2),
                    "PnL %": round((gross_pnl / max(pos["capital_used"], 1e-6)) * 100, 2),
                    "Net PnL %": round((net_pnl / max(pos["capital_used"], 1e-6)) * 100, 2),
                    "Status": "CLOSED",
                })
                executed = "EXITED"
            else:
                reason = "No open SHORT"

        elif signal in ("BUY", "SHORT"):
            if signal == "SHORT" and not ENABLE_SHORTING:
                executed, reason = "SKIPPED", "Shorting disabled"
            elif stock in open_positions:
                executed, reason = "SKIPPED", "Position already open"
            elif stop_new_entries:
                executed, reason = "SKIPPED", "Daily max loss hit"
            elif len(open_positions) >= MAX_OPEN_TRADES:
                executed, reason = "SKIPPED", "Max open trades reached"
            else:
                qty, capital_used, risk_inr = calculate_position_size(price, atr)
                if qty <= 0:
                    executed, reason = "SKIPPED", "Position size below 1"
                else:
                    open_positions[stock] = {
                        "side": "LONG" if signal == "BUY" else "SHORT",
                        "entry_price": price,
                        "entry_time": ts,
                        "Qty": qty,
                        "capital_used": capital_used,
                        "risk_inr": risk_inr,
                    }
                    executed = "ENTERED"

        if realized_pnl_inr <= -(STARTING_CAPITAL * DAILY_MAX_LOSS_PCT):
            stop_new_entries = True

        executed_signal_events.append({
            "Stock": stock,
            "Timestamp": ts,
            "Signal": signal,
            "Signal Side": side,
            "Price": round(price, 2),
            "Confidence %": round(confidence, 1),
            "Supertrend": supertrend,
            "Execution": executed,
            "Reason": reason,
        })

    # Mark-to-market open positions at EOD
    for stock, pos in open_positions.items():
        data = stock_data.get(stock)
        if data is None or data.empty:
            continue
        close_price = float(data["Close"].iloc[-1])
        exit_time = data.index[-1]
        if pos["side"] == "LONG":
            gross_pnl = (close_price - pos["entry_price"]) * pos["Qty"]
        else:
            gross_pnl = (pos["entry_price"] - close_price) * pos["Qty"]
        net_pnl = _net_pnl(gross_pnl, pos["capital_used"])
        trades.append({
            "Stock": stock,
            "Side": pos["side"],
            "Entry Time": pos["entry_time"],
            "Exit Time": exit_time,
            "Qty": pos["Qty"],
            "Entry Price": round(pos["entry_price"], 2),
            "Exit Price": round(close_price, 2),
            "Capital Used": round(pos["capital_used"], 2),
            "Risk at Entry (INR)": round(pos["risk_inr"], 2),
            "Exit Signal": "EOD_MARK_TO_MARKET",
            "Gross PnL (INR)": round(gross_pnl, 2),
            "Costs (INR)": round(gross_pnl - net_pnl, 2),
            "Net PnL (INR)": round(net_pnl, 2),
            "PnL %": round((gross_pnl / max(pos["capital_used"], 1e-6)) * 100, 2),
            "Net PnL %": round((net_pnl / max(pos["capital_used"], 1e-6)) * 100, 2),
            "Status": "OPEN_MTM",
        })

    signal_df = pd.DataFrame(executed_signal_events)
    trades_df = pd.DataFrame(trades)
    return signal_df, trades_df


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report():
    stocks = get_dynamic_stocks(enable_shorting=ENABLE_SHORTING)
    all_signal_events: list[dict] = []
    stock_data: dict[str, pd.DataFrame] = {}

    for stock in stocks:
        try:
            data = fetch_intraday_data(stock)
            if data.empty or len(data) < 35:
                continue
            data = calculate_indicators(data)
            data = generate_signals(data)
            data = data.dropna(subset=["Close", "Signal"])
            if data.empty:
                continue
            stock_data[stock] = data
            all_signal_events.extend(build_signal_events(stock, data))
        except Exception as exc:
            print(f"Error processing {stock}: {exc}")

    signal_df, trades_df = simulate_portfolio_day(all_signal_events, stock_data)
    return stocks, signal_df, trades_df


def print_report(stocks, signal_df: pd.DataFrame, trades_df: pd.DataFrame):
    print("=" * 96)
    print("INTRADAY MARKET CLOSE REPORT  (all P&L figures are NET of transaction costs)")
    print("=" * 96)
    print(f"Selected Universe ({len(stocks)}): {', '.join(stocks)}")
    print(f"Round-trip cost model: {ROUND_TRIP_COST_PCT*100:.4f}% of capital used per trade")
    print("-" * 96)

    if signal_df.empty:
        print("No actionable BUY/SELL-side signals generated today.")
    else:
        print("Signal Events (BUY/SELL side actions):")
        view_cols = [
            "Timestamp", "Stock", "Signal", "Signal Side",
            "Price", "Confidence %", "Supertrend", "Execution", "Reason",
        ]
        available = [c for c in view_cols if c in signal_df.columns]
        print(
            signal_df[available]
            .sort_values(["Timestamp", "Stock"])
            .to_string(index=False)
        )
    print("-" * 96)

    if trades_df.empty:
        print("No trades were opened/closed by the strategy today.")
        print("Day Net PnL %: 0.00")
        return

    view_cols = [
        "Stock", "Side", "Entry Time", "Exit Time", "Qty",
        "Entry Price", "Exit Price", "Capital Used",
        "Risk at Entry (INR)", "Exit Signal",
        "Gross PnL (INR)", "Costs (INR)", "Net PnL (INR)",
        "PnL %", "Net PnL %", "Status",
    ]
    available = [c for c in view_cols if c in trades_df.columns]
    print("Trade Log:")
    print(
        trades_df[available]
        .sort_values(["Exit Time", "Stock"])
        .to_string(index=False)
    )

    total_gross_inr = float(trades_df["Gross PnL (INR)"].sum()) if "Gross PnL (INR)" in trades_df.columns else 0.0
    total_costs_inr = float(trades_df["Costs (INR)"].sum()) if "Costs (INR)" in trades_df.columns else 0.0
    total_net_inr = float(trades_df["Net PnL (INR)"].sum()) if "Net PnL (INR)" in trades_df.columns else 0.0
    total_net_pct = (total_net_inr / STARTING_CAPITAL) * 100
    closed_count = int((trades_df["Status"] == "CLOSED").sum())
    mtm_count = int((trades_df["Status"] == "OPEN_MTM").sum())

    print("-" * 96)
    print(f"Closed Trades: {closed_count}  |  Open (MTM at EOD): {mtm_count}")
    print(f"Day Gross PnL (INR): {total_gross_inr:.2f}")
    print(f"Day Transaction Costs (INR): {total_costs_inr:.2f}")
    print(f"Day Net PnL (INR): {total_net_inr:.2f}")
    print(f"Day Net PnL % (on starting capital ₹{STARTING_CAPITAL:,.0f}): {total_net_pct:.2f}%")
    print(
        f"Risk Limits → max capital/trade: {MAX_CAPITAL_PER_TRADE_PCT*100:.0f}%  |  "
        f"risk/trade: {RISK_PER_TRADE_PCT*100:.0f}%  |  "
        f"max daily loss: {DAILY_MAX_LOSS_PCT*100:.1f}%  |  "
        f"max open trades: {MAX_OPEN_TRADES}"
    )


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------
def to_ist_timestamp(value) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(IST_ZONE).isoformat(sep=" ")


def convert_report_timestamps_to_ist(
    signal_df: pd.DataFrame, trades_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_out = signal_df.copy()
    trades_out = trades_df.copy()
    if "Timestamp" in signal_out.columns:
        signal_out["Timestamp"] = signal_out["Timestamp"].map(to_ist_timestamp)
    for col in ("Entry Time", "Exit Time"):
        if col in trades_out.columns:
            trades_out[col] = trades_out[col].map(to_ist_timestamp)
    return signal_out, trades_out


# ---------------------------------------------------------------------------
# Save to CSV
# ---------------------------------------------------------------------------
def save_report(
    output_dir: str,
    now_ist: datetime.datetime,
    signal_df: pd.DataFrame,
    trades_df: pd.DataFrame,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag = now_ist.strftime("%Y%m%d")

    signal_file = output_path / f"signals_{date_tag}.csv"
    trades_file = output_path / f"trades_{date_tag}.csv"

    signal_df.to_csv(signal_file, index=False)
    trades_df.to_csv(trades_file, index=False)
    print(f"Saved signal report : {signal_file}")
    print(f"Saved trade report  : {trades_file}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate post-market intraday BUY/SELL report with "
            "gross and net P&L (after transaction costs)."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run report even before 15:30 IST or on non-trading days.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory where CSV report files are saved.",
    )
    args = parser.parse_args()

    now_ist = get_ist_now()
    if not args.force:
        if not is_trading_day(now_ist):
            print("Today is not an NSE trading day. Use --force to run anyway.")
            return
        if not is_after_market_close(now_ist):
            print(
                f"Market close report is available after 15:30 IST. "
                f"Current IST time: {now_ist.strftime('%H:%M:%S')}"
            )
            return

    stocks, signal_df, trades_df = generate_report()
    signal_df, trades_df = convert_report_timestamps_to_ist(signal_df, trades_df)
    print_report(stocks, signal_df, trades_df)
    save_report(args.output_dir, now_ist, signal_df, trades_df)


if __name__ == "__main__":
    main()
