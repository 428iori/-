# -*- coding: utf-8 -*-
"""
自動株式売買システム (完全版)
・日次実行 (GitHub Actions/Colab対応)
・前回残高・保有株を equity_state.json から引き継ぎ
・held_daysを1日進め、自動売却
・新規買付を自動実行
・trade_history.csv, equity_log.csv, equity_state.json を更新
・Discord通知に対応
"""

import os, json, csv, datetime as dt
import pandas as pd
import yfinance as yf
import requests

# ===== 設定 =====
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
CAPITAL_LIMIT = 1_000_000
PER_POS_FRAC = 0.20
HOLD_DAYS_DEFAULT = 5
TICKERS_TODAY = ["NVDA", "PYPL", "NKE"]  # テスト用。実際はモデル出力に置換。

# ======== 共通関数 ========

def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default or {}

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def append_csv(path, row, header):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def discord_notify(msg: str):
    if DISCORD_WEBHOOK_URL:
        try:
            requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
        except Exception as e:
            print("[WARN] Discord通知失敗:", e)

def get_market_price(ticker, field="Close"):
    try:
        data = yf.download(ticker, period="2d", progress=False)
        if len(data) > 0:
            return float(data[field].iloc[-1])
    except Exception:
        pass
    return None

# ======== 売買管理 ========

def advance_positions_one_day(state):
    """held_daysを進め、売却対象を検出"""
    new_positions, sold_positions = [], []
    for pos in state.get("positions", []):
        pos["held_days"] += 1
        if pos["held_days"] >= pos["hold_days"]:
            sold_positions.append(pos)
        else:
            new_positions.append(pos)
    state["positions"] = new_positions
    return state, sold_positions

def sell_positions(state, sold_positions):
    """売却処理（単純に現価格で決済）"""
    realized_pnl = 0.0
    sell_rows = []
    for pos in sold_positions:
        px = get_market_price(pos["ticker"], "Close") or pos["buy_price"]
        pnl = (px - pos["buy_price"]) * (pos["amount"] / pos["buy_price"])
        realized_pnl += pnl
        sell_rows.append({
            "date": dt.date.today().isoformat(),
            "action": "SELL",
            "ticker": pos["ticker"],
            "price": round(px, 2),
            "amount": pos["amount"],
            "pnl": round(pnl, 2),
            "hold_days": pos["hold_days"],
            "held_days": pos["held_days"],
            "src": "auto_close"
        })
    state["capital"] += realized_pnl
    return state, realized_pnl, sell_rows

def buy_new_positions(state, tickers):
    buys = []
    available = state["capital"] * PER_POS_FRAC
    for tkr in tickers:
        price = get_market_price(tkr, "Close")
        if not price:
            print(f"[WARN] {tkr}: 価格取得失敗")
            continue
        amt = available
        pos = {
            "ticker": tkr,
            "buy_price": price,
            "amount": amt,
            "hold_days": HOLD_DAYS_DEFAULT,
            "held_days": 0
        }
        state["positions"].append(pos)
        state["capital"] -= amt
        buys.append({
            "date": dt.date.today().isoformat(),
            "action": "BUY",
            "ticker": tkr,
            "price": round(price, 2),
            "amount": amt,
            "pnl": "",
            "hold_days": HOLD_DAYS_DEFAULT,
            "held_days": 0,
            "src": "te_close(today)"
        })
    return state, buys

# ======== メイン処理 ========

def main():
    run_date = dt.date.today().isoformat()
    market_date = (dt.date.today() - dt.timedelta(days=1)).isoformat()

    # --- ファイル読込 ---
    state = load_json("equity_state.json", {"capital": CAPITAL_LIMIT, "positions": []})

    # --- 1日進行 & 売却 ---
    state, sold_positions = advance_positions_one_day(state)
    state, realized_pnl, sell_rows = sell_positions(state, sold_positions) if sold_positions else (state, 0.0, [])

    # --- 新規購入 ---
    state, buy_rows = buy_new_positions(state, TICKERS_TODAY)

    # --- 未実現損益計算 ---
    unrealized = 0.0
    for p in state["positions"]:
        price = get_market_price(p["ticker"], "Close") or p["buy_price"]
        unrealized += (price - p["buy_price"]) * (p["amount"] / p["buy_price"])
    equity = state["capital"] + unrealized

    # --- ログ出力 ---
    equity_row = {
        "run_date_jst": run_date,
        "market_date": market_date,
        "status": "ok",
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(unrealized, 2),
        "equity": round(equity, 2),
        "capital": round(state["capital"], 2),
        "num_positions": len(state["positions"]),
        "positions": ";".join([p["ticker"] for p in state["positions"]]),
        "buys": ";".join([b["ticker"] for b in buy_rows]),
        "sells": ";".join([s["ticker"] for s in sell_rows]),
        "daily_return_pct": round(unrealized / max(1, state["capital"]), 4),
        "error": ""
    }

    append_csv("equity_log.csv", equity_row, list(equity_row.keys()))
    for row in (buy_rows + sell_rows):
        append_csv("trade_history.csv", row, list(row.keys()))

    save_json("equity_state.json", state)

    # --- Discord通知 ---
    msg = (
        f"📅 **{run_date} トレード結果**\n"
        f"**市場データ日:** {market_date}\n"
        f"**売却:** {', '.join([s['ticker'] for s in sell_rows]) if sell_rows else '（売却なし）'}\n\n"
        f"**新規購入:** {', '.join([b['ticker'] for b in buy_rows]) if buy_rows else '（購入なし）'}\n\n"
        f"**実現損益:** {realized_pnl:+.0f}円\n"
        f"**含み損益:** {unrealized:+.0f}円\n"
        f"**評価残高:** {equity:,.0f}円\n"
        f"**確定残高:** {state['capital']:,.0f}円\n"
        f"**ポジション数:** {len(state['positions'])}"
    )
    discord_notify(msg)

    print(msg)
    print("[✅] ファイル更新完了:", run_date)

if __name__ == "__main__":
    main()
