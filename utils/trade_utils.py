import pandas as pd
from utils.file_io import append_csv_row

def execute_buy(ticker, price, amount, hold_days, capital, positions, log_path, market_date):
    cost = amount
    if cost > capital:
        print(f"[skip] insufficient funds for {ticker}")
        return capital, positions

    capital -= cost
    positions.append({
        "ticker": ticker,
        "buy_price": price,
        "amount": amount,
        "hold_days": hold_days,
        "held_days": 0
    })
    print(f"[buy] {ticker} {amount:.0f}円 @ {price:.2f}")

    append_csv_row(log_path, {
        "date": market_date,
        "action": "BUY",
        "ticker": ticker,
        "price": price,
        "amount": amount
    })
    return capital, positions

def execute_sell(pos, sell_price, capital, log_path, market_date):
    pnl = (sell_price - pos["buy_price"]) / pos["buy_price"] * pos["amount"]
    capital += pos["amount"] + pnl
    append_csv_row(log_path, {
        "date": market_date,
        "action": "SELL",
        "ticker": pos["ticker"],
        "price": sell_price,
        "amount": pos["amount"],
        "pnl": pnl
    })
    print(f"[sell] {pos['ticker']} @ {sell_price:.2f} → pnl={pnl:.0f}")
    return capital, pnl

def evaluate_positions(positions, market_date, capital):
    unreal = 0
    for pos in positions:
        current_price = pos["buy_price"]  # 実運用ではfetch価格
        unreal += (current_price - pos["buy_price"]) / pos["buy_price"] * pos["amount"]
    total_value = capital + sum(p["amount"] for p in positions)
    return unreal, total_value
