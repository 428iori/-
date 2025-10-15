# -*- coding: utf-8 -*-
import os, json, pandas as pd
from datetime import datetime
from utils.data_fetch import get_execution_price_for_sell, get_execution_price_for_buy
from utils.file_io import load_equity_state, save_equity_state, append_csv_row
from utils.trade_utils import execute_buy, execute_sell, evaluate_positions
from utils.discord_notify import send_discord_message

INIT_CAPITAL = 1_000_000
STATE_FILE = "equity_state.json"
DAILY_LOG = "equity_log.csv"
TRADE_LOG = "trade_history.csv"

def today_str():
    return datetime.now().strftime("%Y-%m-%d")

def main():
    market_date = today_str()
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # === 状態ロード ===
    state = load_equity_state(STATE_FILE, INIT_CAPITAL)
    capital = state["capital"]
    positions = state["positions"]
    realized_pnl = 0.0

    print(f"\n📅 [{market_date}] start trading | capital={capital:,.0f} | pos={len(positions)}")

    # === 売却チェック ===
    new_positions = []
    for pos in positions:
        pos["held_days"] += 1
        if pos["held_days"] >= pos["hold_days"]:
            sell_px = get_execution_price_for_sell(pos["ticker"], market_date)
            capital, pnl = execute_sell(pos, sell_px, capital, TRADE_LOG, market_date)
            realized_pnl += pnl
        else:
            new_positions.append(pos)
    positions = new_positions

    # === 新規買付（例） ===
    buy_signals = [
        {"ticker": "QCOM", "price": 161.77, "hold_days": 5},
        {"ticker": "BA", "price": 214.34, "hold_days": 5},
        {"ticker": "FDX", "price": 234.72, "hold_days": 5},
    ]

    per_pos_frac = 0.2  # 20%資金投入上限
    for sig in buy_signals:
        amount = capital * per_pos_frac / len(buy_signals)
        capital, positions = execute_buy(
            sig["ticker"], sig["price"], amount, sig["hold_days"], capital, positions, TRADE_LOG, market_date
        )

    # === 評価額算出 ===
    unrealized_pnl, total_value = evaluate_positions(positions, market_date, capital)
    equity = capital + unrealized_pnl

    # === 日次ログ ===
    append_csv_row(DAILY_LOG, {
        "run_date_jst": run_time,
        "market_date": market_date,
        "capital": capital,
        "equity": equity,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "num_positions": len(positions)
    })

    # === 状態保存 ===
    save_equity_state(STATE_FILE, {"capital": capital, "positions": positions})

    # === Discord通知 ===
    msg = (
        f"📅 **{market_date} トレード結果**\n"
        f"**売却:** {'（なし）' if realized_pnl == 0 else f'+{realized_pnl:.0f}円'}\n"
        f"**新規購入:** {', '.join(p['ticker'] for p in positions[-len(buy_signals):])}\n"
        f"**残高:** {capital:,.0f}円\n"
        f"**評価額:** {equity:,.0f}円\n"
        f"**ポジション:** {len(positions)}銘柄"
    )
    send_discord_message(msg)

if __name__ == "__main__":
    main()
