# run_production_day.py
import json, csv, os, datetime as dt
import pandas as pd
import lightgbm as lgb
from utils_data import load_recent_data
import requests

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
CAPITAL_LIMIT = 1_000_000
TOP_K = 3
PER_POS_FRAC = 0.20
HOLD_DAYS_DEFAULT = 5
MODEL_PATH = "model_best.lgb"

def discord_notify(msg):
    if not DISCORD_WEBHOOK_URL: return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
    except Exception as e:
        print("[WARN] Discord通知失敗:", e)

def load_state():
    if os.path.exists("equity_state.json"):
        return json.load(open("equity_state.json", "r"))
    return {"capital": CAPITAL_LIMIT, "positions": []}

def save_state(state):
    json.dump(state, open("equity_state.json", "w"), indent=2, ensure_ascii=False)

def append_csv(path, row):
    header = list(row.keys())
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists: writer.writeheader()
        writer.writerow(row)

def main():
    today = dt.date.today().isoformat()
    market_date = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    state = load_state()

    # === LightGBM モデル読み込み ===
    model = lgb.Booster(model_file=MODEL_PATH)

    # === 特徴量生成 ===
    tickers = ["AAPL","MSFT","NVDA","TSLA","AMZN","QCOM","META","BA","FDX","NKE","PYPL"]
    df = load_recent_data(tickers, start="2024-01-01")
    df_last = df.groupby("ticker").tail(1).reset_index(drop=True)
    feats = [c for c in df_last.columns if c not in ["ticker","Date"]]
    df_last["p_up"] = model.predict(df_last[feats])

    # === スコア選定 ===
    df_last["score"] = df_last["p_up"] * 1.0  # μ固定 (32%時の設計)
    top = df_last.sort_values("score", ascending=False).head(TOP_K)
    tickers_today = top["ticker"].tolist()

    # === 売却 & 保有日更新 ===
    new_positions, sold_positions = [], []
    realized = 0
    for p in state["positions"]:
        p["held_days"] += 1
        if p["held_days"] >= p["hold_days"]:
            sold_positions.append(p)
        else:
            new_positions.append(p)
    state["positions"] = new_positions

    # === 売却処理 ===
    for s in sold_positions:
        px_sell = df_last.loc[df_last["ticker"]==s["ticker"], "Close"].values[0]
        pnl = (px_sell - s["buy_price"]) * (s["amount"] / s["buy_price"])
        state["capital"] += s["amount"] + pnl
        realized += pnl
        append_csv("trade_history.csv", {
            "date": today, "action": "SELL", "ticker": s["ticker"],
            "price": round(px_sell,2), "amount": s["amount"], "pnl": round(pnl,2),
            "hold_days": s["hold_days"], "held_days": s["held_days"], "src": "auto"
        })

    # === 新規購入 ===
    buys = []
    available = state["capital"] * PER_POS_FRAC
    for tkr in tickers_today:
        px = df_last.loc[df_last["ticker"]==tkr, "Close"].values[0]
        amt = available
        state["capital"] -= amt
        buys.append({
            "ticker": tkr, "buy_price": px, "amount": amt,
            "hold_days": HOLD_DAYS_DEFAULT, "held_days": 0
        })
        append_csv("trade_history.csv", {
            "date": today, "action": "BUY", "ticker": tkr,
            "price": round(px,2), "amount": amt, "pnl": "",
            "hold_days": HOLD_DAYS_DEFAULT, "held_days": 0, "src": "pred"
        })

    state["positions"].extend(buys)
    unreal = 0
    for p in state["positions"]:
        cur_px = df_last.loc[df_last["ticker"]==p["ticker"], "Close"].values[0]
        unreal += (cur_px - p["buy_price"]) * (p["amount"]/p["buy_price"])
    equity = state["capital"] + unreal

    # === ログ出力 ===
    row = {
        "run_date_jst": today,
        "market_date": market_date,
        "status": "ok",
        "realized_pnl": round(realized, 2),
        "unrealized_pnl": round(unreal, 2),
        "equity": round(equity, 2),
        "capital": round(state["capital"], 2),
        "num_positions": len(state["positions"]),
        "positions": ";".join([p["ticker"] for p in state["positions"]]),
        "buys": ";".join([b["ticker"] for b in buys]),
        "sells": ";".join([s["ticker"] for s in sold_positions]),
        "daily_return_pct": round(unreal/max(1,state["capital"]),4),
        "error": ""
    }
    append_csv("equity_log.csv", row)
    save_state(state)

    msg = (
        f"📅 **{today} トレード結果**\n"
        f"**売却:** {', '.join([s['ticker'] for s in sold_positions]) if sold_positions else '（売却なし）'}\n"
        f"**新規購入:** {', '.join(tickers_today)}\n"
        f"**実現損益:** {realized:+.0f}円\n"
        f"**含み損益:** {unreal:+.0f}円\n"
        f"**評価残高:** {equity:,.0f}円\n"
        f"**確定残高:** {state['capital']:,.0f}円"
    )
    discord_notify(msg)
    print(msg)

if __name__ == "__main__":
    main()
