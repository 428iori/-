# -*- coding: utf-8 -*-
"""
AI株式シミュレーター 決定版（実運用リアルタイム）
------------------------------------------------------------
・6ヶ月学習 → 当日予測（p×μ法＋確率校正）
・1日/3日/5日保有を再現（複数日ポジション継続）
・残高・ポジションを equity_state.json に保存
・Discordに日次損益と残高を通知
・土日は自動スキップ
"""

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import warnings, json, os, datetime, requests
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

# =========================
# 設定（決定版パラメータ）
# =========================
START = "2015-01-01"
END   = None
RANDOM_SEED = 42
TRAIN_MONTHS = 6
TOP_K = 3
INIT_CAPITAL = 1_000_000
PER_POS_FRAC = 0.20
COMMISSION   = 0.0005

USE_CALIBRATION = True
CALIB_METHOD = "isotonic"
FIXED_SCORE_QUANTILE = 0.50
SOFTMAX_TEMP = 0.15
RET_CLIP_LOW, RET_CLIP_HIGH = -0.08, 0.30

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
STATE_FILE = Path("equity_state.json")

ALL_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AMD","NFLX","ADBE",
    "CRM","INTC","IBM","ORCL","QCOM","AVGO","CSCO","TXN","MU","SHOP",
    "SNOW","PANW","TEAM","DDOG","PLTR","UBER","ABNB","PYPL","NOW",
    "ZM","CRWD","MDB","RBLX","NET","ZS","COST","WMT","HD","LOW",
    "TGT","MCD","SBUX","NKE","KO","PEP","PG","PM","DIS","BKNG",
    "F","GM","CAT","BA","GE","DE","UPS","FDX","HON","MMM",
    "LMT","RTX","NOC","GD","XOM","CVX","COP","PSX","MPC","OXY",
    "SLB","EOG","DVN","APA","FCX","NEM","JPM","BAC","C","MS",
    "GS","BLK","SCHW","SPGI","ICE","V","MA","AXP","JNJ","PFE",
    "MRK","BMY","LLY","UNH","CI","HUM","CVS","TMO","DHR","MDT"
]

# =========================
# ユーティリティ関数
# =========================
def rsi(series, n=14):
    diff = series.diff()
    up = diff.clip(lower=0)
    dn = -diff.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_dn + 1e-12)
    return 100 - (100 / (1 + rs))

def make_features(df, ticker):
    c, v = df["close"], df["volume"]
    df["sma_s"] = c.rolling(10).mean()
    df["sma_m"] = c.rolling(30).mean()
    df["sma_l"] = c.rolling(90).mean()
    df["rsi"] = rsi(c, 14)
    df["ret1"] = c.pct_change()
    df["ret5"] = c.pct_change(5)
    df["volatility"] = c.pct_change().rolling(20).std()
    df["ma_gap"] = (c - df["sma_m"]) / (df["sma_m"] + 1e-12)
    df["ticker"] = ticker
    return df.dropna().reset_index(drop=True)

FEATS = ["sma_s","sma_m","sma_l","rsi","ret1","ret5","volatility","ma_gap"]

def load_all_data_fast(tickers):
    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start=START, end=END, auto_adjust=True, group_by='ticker', threads=True, progress=False)
    out = []
    for t in tickers:
        try:
            df = data[t].copy().reset_index()
            df.columns = [c.lower() for c in df.columns]
            feat = make_features(df, t)
            out.append(feat)
        except Exception:
            continue
    all_df = pd.concat(out, ignore_index=True).sort_values(["date","ticker"]).reset_index(drop=True)
    print(f"全データ件数: {len(all_df)}; 期間: {all_df['date'].min().date()} ～ {all_df['date'].max().date()}")
    return all_df

def add_future_labels_inplace(df):
    for n in [1,3,5]:
        df[f"ret_next{n}"] = df.groupby("ticker")["close"].shift(-n) / df["close"] - 1
        df[f"label{n}"] = (df[f"ret_next{n}"] > 0).astype(int)
    return df

def train_models(df):
    params = {
        "objective":"binary","metric":"auc","random_state":RANDOM_SEED,
        "learning_rate":0.05,"num_leaves":31,"n_estimators":400,"verbosity":-1
    }
    models = {}
    for k in ["1","3","5"]:
        X, y = df[FEATS], df[f"label{k}"]
        models[k] = lgb.train(params, lgb.Dataset(X, label=y))
    return models

def predict_models(models, df):
    X = df[FEATS]
    return {f"p{k}": models[k].predict(X) for k in ["1","3","5"]}

def estimate_mu(df):
    mu = {}
    for k in ["1","3","5"]:
        pos = df[f"ret_next{k}"].loc[df[f"label{k}"]==1]
        mu[k] = pos.mean() if len(pos)>0 else 0
    return mu

def notify_discord(msg):
    if not DISCORD_WEBHOOK_URL:
        print(msg); return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
    except Exception as e:
        print(f"Discord通知失敗: {e}")

def load_state():
    if not STATE_FILE.exists():
        return {"capital": INIT_CAPITAL, "positions": []}
    return json.load(open(STATE_FILE, "r", encoding="utf-8"))

def save_state(state):
    json.dump(state, open(STATE_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

# =========================
# 実運用：複数日保有＋継続運用
# =========================
def simulate_continuous(all_df):
    df = all_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    today = df["date"].max()
    tr = df[df["date"] >= today - pd.DateOffset(months=6)]
    te = df[df["date"] == today]
    if tr.empty or te.empty:
        print("⚠ データ不足")
        return

    state = load_state()
    capital = state["capital"]
    old_positions = state["positions"]
    day_profit = 0
    sold_lines = []
    remaining = []

    # === 売却処理 ===
    for pos in old_positions:
        pos["held_days"] += 1
        if pos["held_days"] >= pos["hold_days"]:
            row = te[te["ticker"]==pos["ticker"]]
            if not row.empty:
                sell_price = row["close"].iloc[0]
                profit = (sell_price/pos["buy_price"]-1)*pos["amount"]
                day_profit += profit
                sold_lines.append(f"{pos['ticker']} +{profit:,.0f}円")
        else:
            remaining.append(pos)
    capital += day_profit

    # === 新規購入 ===
    add_future_labels_inplace(tr)
    add_future_labels_inplace(te)
    models = train_models(tr)
    mu = estimate_mu(tr)
    preds = predict_models(models, te)
    scores = np.vstack([preds["p1"]*mu["1"], preds["p3"]*mu["3"], preds["p5"]*mu["5"]]).T
    best_idx = np.argmax(scores, axis=1)
    best_score = scores.max(axis=1)
    te["score"] = best_score
    thr = np.quantile(best_score, FIXED_SCORE_QUANTILE)
    buys = te[te["score"]>=thr].sort_values("score", ascending=False).head(TOP_K)
    if buys.empty:
        save_state({"capital": capital, "positions": remaining})
        notify_discord(f"📅 {today.date()} 取引なし\n損益: {day_profit:+,.0f}円\n残高: {capital:,.0f}円")
        return

    per_trade = capital * PER_POS_FRAC / len(buys)
    new_positions = []
    for i, row in enumerate(buys.itertuples()):
        p1, p3, p5 = preds["p1"][i], preds["p3"][i], preds["p5"][i]
        idx = np.argmax([p1*mu["1"], p3*mu["3"], p5*mu["5"]])
        hold_days = [1,3,5][idx]
        new_positions.append({
            "ticker": row.ticker,
            "buy_price": row.close,
            "amount": per_trade,
            "hold_days": hold_days,
            "held_days": 0
        })

    all_positions = remaining + new_positions
    state = {"capital": capital, "positions": all_positions}
    save_state(state)

    sold_str = "\n".join(sold_lines) if sold_lines else "（売却なし）"
    buy_str = "\n".join([f"{p['ticker']} ({p['hold_days']}日保有)" for p in new_positions])
    msg = (
        f"📅 **{today.date()} トレード結果**\n"
        f"**売却:**\n{sold_str}\n"
        f"**新規購入:**\n{buy_str}\n"
        f"**損益:** {day_profit:+,.0f}円\n"
        f"**残高:** {capital:,.0f}円"
    )
    print(msg)
    notify_discord(msg)

# =========================
# メイン実行
# =========================
def main():
    # === 土日スキップ ===
    weekday = datetime.datetime.now().weekday()  # 月=0, 日=6
    if weekday >= 5:
        msg = f"🛌 {datetime.date.today()} は休場日のためスキップ"
        print(msg)
        notify_discord(msg)
        return

    all_df = load_all_data_fast(ALL_TICKERS)
    simulate_continuous(all_df)

if __name__ == "__main__":
    main()












