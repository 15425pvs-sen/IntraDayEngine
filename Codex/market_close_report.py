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
    STARTING_CAPITAL,
    STOP_LOSS_ATR_MULTIPLIER,
    calculate_indicators,
    ENABLE_SHORTING,
    fetch_intraday_data,
    generate_signals,
    get_dynamic_stocks,
    get_ist_now,
    is_nse_trading_holiday,
)


def is_after_market_close(now_ist):
    return now_ist.time() >= datetime.time(15, 30)


def is_trading_day(now_ist):
    if now_ist.weekday() >= 5:
        return False
    if is_nse_trading_holiday(now_ist.date()):
        return False
    return True


def signal_side(signal):
    if signal in ("BUY", "COVER"):
        return "BUY"
    if signal in ("SHORT", "EXIT_LONG", "SELL"):
        return "SELL"
    return "HOLD"


def build_signal_events(stock, data):
    rows = []
    for ts, row in data.iterrows():
        signal = row["Signal"]
        if signal == "HOLD":
            continue
        price = float(row["Close"])
        confidence = float(row.get("Signal_Confidence", 0.0))
        atr = float(row.get("ATR", 0.0)) if pd.notna(row.get("ATR", 0.0)) else 0.0
        rows.append(
            {
                "Stock": stock,
                "Timestamp": ts,
                "Signal": signal,
                "Signal Side": signal_side(signal),
                "Price": price,
                "Confidence %": confidence,
                "ATR": atr,
            }
        )
    return rows


def calculate_position_size(price, atr_value):
    max_capital = STARTING_CAPITAL * MAX_CAPITAL_PER_TRADE_PCT
    risk_cash = STARTING_CAPITAL * RISK_PER_TRADE_PCT
    stop_distance = max((atr_value * STOP_LOSS_ATR_MULTIPLIER), price * MIN_STOP_LOSS_PCT)
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


def simulate_portfolio_day(signal_events, stock_data):
    signal_events = sorted(signal_events, key=lambda x: (x["Timestamp"], x["Stock"]))
    executed_signal_events = []
    trades = []

    open_positions = {}
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

        executed = "IGNORED"
        reason = ""

        if signal in ("EXIT_LONG", "SELL"):
            if stock in open_positions and open_positions[stock]["side"] == "LONG":
                pos = open_positions.pop(stock)
                pnl_inr = (price - pos["entry_price"]) * pos["Qty"]
                realized_pnl_inr += pnl_inr
                trades.append(
                    {
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
                        "PnL (INR)": round(pnl_inr, 2),
                        "PnL %": round((pnl_inr / max(pos["capital_used"], 1e-6)) * 100, 2),
                        "Status": "CLOSED",
                    }
                )
                executed = "EXITED"
            else:
                reason = "No open LONG"
        elif signal == "COVER":
            if stock in open_positions and open_positions[stock]["side"] == "SHORT":
                pos = open_positions.pop(stock)
                pnl_inr = (pos["entry_price"] - price) * pos["Qty"]
                realized_pnl_inr += pnl_inr
                trades.append(
                    {
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
                        "PnL (INR)": round(pnl_inr, 2),
                        "PnL %": round((pnl_inr / max(pos["capital_used"], 1e-6)) * 100, 2),
                        "Status": "CLOSED",
                    }
                )
                executed = "EXITED"
            else:
                reason = "No open SHORT"
        elif signal in ("BUY", "SHORT"):
            if signal == "SHORT" and not ENABLE_SHORTING:
                executed = "SKIPPED"
                reason = "Shorting disabled"
            elif stock in open_positions:
                executed = "SKIPPED"
                reason = "Position already open"
            elif stop_new_entries:
                executed = "SKIPPED"
                reason = "Daily max loss hit"
            elif len(open_positions) >= MAX_OPEN_TRADES:
                executed = "SKIPPED"
                reason = "Max open trades reached"
            else:
                qty, capital_used, risk_inr = calculate_position_size(price, atr)
                if qty <= 0:
                    executed = "SKIPPED"
                    reason = "Position size below 1"
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

        executed_signal_events.append(
            {
                "Stock": stock,
                "Timestamp": ts,
                "Signal": signal,
                "Signal Side": side,
                "Price": round(price, 2),
                "Confidence %": round(confidence, 1),
                "Execution": executed,
                "Reason": reason,
            }
        )

    for stock, pos in open_positions.items():
        data = stock_data.get(stock)
        if data is None or data.empty:
            continue
        close_price = float(data["Close"].iloc[-1])
        exit_time = data.index[-1]
        if pos["side"] == "LONG":
            pnl_inr = (close_price - pos["entry_price"]) * pos["Qty"]
        else:
            pnl_inr = (pos["entry_price"] - close_price) * pos["Qty"]
        trades.append(
            {
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
                "PnL (INR)": round(pnl_inr, 2),
                "PnL %": round((pnl_inr / max(pos["capital_used"], 1e-6)) * 100, 2),
                "Status": "OPEN_MTM",
            }
        )

    signal_df = pd.DataFrame(executed_signal_events)
    trades_df = pd.DataFrame(trades)
    return signal_df, trades_df


def generate_report():
    stocks = get_dynamic_stocks(enable_shorting=ENABLE_SHORTING)
    all_signal_events = []
    stock_data = {}

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


def print_report(stocks, signal_df, trades_df):
    print("=" * 88)
    print("INTRADAY MARKET CLOSE REPORT")
    print("=" * 88)
    print(f"Selected Universe ({len(stocks)}): {', '.join(stocks)}")
    print("-" * 88)

    if signal_df.empty:
        print("No actionable BUY/SELL-side signals generated today.")
    else:
        print("Signal Events (BUY/SELL side actions):")
        view_cols = [
            "Timestamp",
            "Stock",
            "Signal",
            "Signal Side",
            "Price",
            "Confidence %",
            "Execution",
            "Reason",
        ]
        print(signal_df[view_cols].sort_values(["Timestamp", "Stock"]).to_string(index=False))
    print("-" * 88)

    if trades_df.empty:
        print("No trades were opened/closed by the strategy today.")
        print("Day PnL %: 0.00")
        return

    view_cols = [
        "Stock",
        "Side",
        "Entry Time",
        "Exit Time",
        "Qty",
        "Entry Price",
        "Exit Price",
        "Capital Used",
        "Risk at Entry (INR)",
        "Exit Signal",
        "PnL (INR)",
        "PnL %",
        "Status",
    ]
    print("Trade Log:")
    print(trades_df[view_cols].sort_values(["Exit Time", "Stock"]).to_string(index=False))

    total_pnl_inr = float(trades_df["PnL (INR)"].sum()) if "PnL (INR)" in trades_df.columns else 0.0
    total_pnl_pct = (total_pnl_inr / STARTING_CAPITAL) * 100
    closed_count = int((trades_df["Status"] == "CLOSED").sum())
    mtm_count = int((trades_df["Status"] == "OPEN_MTM").sum())
    print("-" * 88)
    print(f"Closed Trades: {closed_count} | Open (MTM at EOD): {mtm_count}")
    print(f"Day PnL (INR): {total_pnl_inr:.2f}")
    print(f"Day PnL % (on starting capital): {total_pnl_pct:.2f}")
    print(
        f"Risk Limits -> Max capital/trade: {MAX_CAPITAL_PER_TRADE_PCT*100:.0f}% | "
        f"Risk/trade: {RISK_PER_TRADE_PCT*100:.0f}% | "
        f"Max daily loss: {DAILY_MAX_LOSS_PCT*100:.1f}% | "
        f"Max open trades: {MAX_OPEN_TRADES}"
    )


def to_ist_timestamp(value):
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return value
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(IST_ZONE).isoformat(sep=" ")


def convert_report_timestamps_to_ist(signal_df, trades_df):
    signal_out = signal_df.copy()
    trades_out = trades_df.copy()

    if "Timestamp" in signal_out.columns:
        signal_out["Timestamp"] = signal_out["Timestamp"].map(to_ist_timestamp)
    for col in ("Entry Time", "Exit Time"):
        if col in trades_out.columns:
            trades_out[col] = trades_out[col].map(to_ist_timestamp)

    return signal_out, trades_out


def save_report(output_dir, now_ist, signal_df, trades_df):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag = now_ist.strftime("%Y%m%d")

    signal_file = output_path / f"signals_{date_tag}.csv"
    trades_file = output_path / f"trades_{date_tag}.csv"

    signal_df.to_csv(signal_file, index=False)
    trades_df.to_csv(trades_file, index=False)
    print(f"Saved signal report: {signal_file}")
    print(f"Saved trade report:  {trades_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate post-market intraday BUY/SELL report with day PnL %."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run report even before 3:30 PM IST or on non-trading days.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory where CSV report files are saved.",
    )
    args = parser.parse_args()

    now_ist = get_ist_now()
    trading_day = is_trading_day(now_ist)
    after_close = is_after_market_close(now_ist)

    if not args.force:
        if not trading_day:
            print("Today is not an NSE trading day. Use --force to run anyway.")
            return
        if not after_close:
            print(
                f"Market close report is available after 15:30 IST. Current IST time: {now_ist.strftime('%H:%M:%S')}"
            )
            return

    stocks, signal_df, trades_df = generate_report()
    signal_df, trades_df = convert_report_timestamps_to_ist(signal_df, trades_df)
    print_report(stocks, signal_df, trades_df)
    save_report(args.output_dir, now_ist, signal_df, trades_df)


if __name__ == "__main__":
    main()
