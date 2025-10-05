# -*- coding: utf-8 -*-
"""
実運用版 AI株式シミュレーター起動スクリプト（Discord通知付き）
"""

import yfinance as yf
import lightgbm as lgb
import pandas as pd
import numpy as np
import requests
import datetime
from datetime import timedelta
import pytz

CFG = {"TIMEZONE": "Asia/Tokyo"}

def main():
    TZ = pytz.timezone(CFG["TIMEZONE"])
    now = datetime.datetime.now(TZ)
    msg = f"🚀 実運用AIシミュレーターを起動しました。\n時刻: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    print(msg)

if __name__ == "__main__":
    main()



# ====== CONFIG ======
CFG = {
    # 基本
    "START": "2015-01-01",
    "END": None,
    "TIMEZONE": "Asia/Tokyo",
    "RANDOM_SEED": 42,

    # 運用パラメータ（発注）
    "INIT_CAPITAL": 1_000_000,     # 運用資金の基準（配分計算用）
    "DAILY_CAPITAL": 1_000_000,    # その日の実際の投下資金（変更可）
    "MAX_GROSS_EXPOSURE": 1.00,    # 合計比率の上限（<=1.0）
    "COMMISSION": 0.0005,          # 往復手数料の片道分
    "SLIPPAGE": 0.0005,            # スリッページ（片道）
    "POSITION_CAP_PCT": 0.35,      # 1銘柄の上限比率（35%）

    # モデル・特徴量
    "TRAIN_MONTHS": 6,             # 学習窓
    "HOLD_MODE": "1d",             # 翌日用シグナル想定
    "LGBM": {"learning_rate":0.03, "num_leaves":31, "n_estimators":200, "max_depth":6},

    # 銘柄選定ロジック
    "TOP_K": 3,                    # 最大採用銘柄数
    "PROB_TH": 0.60,               # 参戦しきい値（平均がこれ未満ならノーポジ）
    "SOFTMAX_TEMP": 0.15,          # 温度パラメータ（小さいほど集中）
    "MIN_WEIGHT_CUT": 0.00,        # 極小ウェイト切り捨て

    # リスク制御
    "RET_CLIP_LOW": -0.10,         # 想定下限（ストレステスト/配分計算の参考）
    "RET_CLIP_HIGH": 0.40,         # 想定上限
    "MAX_PER_SECTOR": 1,           # 同一セクター最大銘柄数（Noneで無効）
    "SECTOR_MAP_CSV": None,        # "sectors.csv" を与えると適用（columns: ticker,sector）

    # Discord通知（任意）
    "DISCORD_WEBHOOK": "DISCORD_WEBHOOK_URL",         # 例: "https://discord.com/api/webhooks/xxxx"
    
    # 成果物保存
    "OUT_DIR": "prod_outputs",     # 生成物フォルダ
    "SAVE_EQUITY_IMG": True,
    "SAVE_SIGNALS_CSV": True,
    "SAVE_ORDERS_CSV": True,
    "SAVE_SUMMARY_JSON": True,
}

np.random.seed(CFG["RANDOM_SEED"])
TZ = timezone(CFG["TIMEZONE"])
os.makedirs(CFG["OUT_DIR"], exist_ok=True)

# ====== UNIVERSE ======
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

# ====== FEATURES ======
def rsi(series: pd.Series, n=14):
    diff = series.diff()
    up, dn = diff.clip(lower=0), -diff.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_dn + 1e-12)
    return 100 - (100/(1+rs))

def make_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    c, v = df["close"].astype(float), df["volume"].astype(float)
    df["sma_s"] = c.rolling(10).mean()
    df["sma_m"] = c.rolling(30).mean()
    df["sma_l"] = c.rolling(90).mean()
    df["rsi"]   = rsi(c, 14)
    df["ret1"]  = c.pct_change()
    df["ret5"]  = c.pct_change(5)
    df["ret10"] = c.pct_change(10)
    df["vol_chg"] = v.pct_change()
    df["ma_gap"]  = (c - df["sma_m"]) / (df["sma_m"] + 1e-12)
    df["volatility"] = c.pct_change().rolling(20).std()
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    df["bb_upper"] = (c - (ma20 + 2*sd20)) / (2*sd20 + 1e-12)
    df["bb_lower"] = (c - (ma20 - 2*sd20)) / (2*sd20 + 1e-12)
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    df["macd"], df["macd_signal"] = macd, macd - macd.ewm(span=9).mean()
    df["vol_ma_ratio"] = v / (v.rolling(20).mean() + 1e-12)
    df["momentum20"]   = c / c.shift(20) - 1
    df["atr"]          = (df["high"] - df["low"]).rolling(14).mean()
    df["dd_from_high"] = c / c.rolling(20).max() - 1
    df["ticker"] = ticker
    return df

FEATS = [
    "sma_s","sma_m","sma_l","rsi","ret1","ret5","ret10","vol_chg",
    "ma_gap","volatility","bb_upper","bb_lower","macd","macd_signal",
    "spy_ret","qqq_ret","vix_ret","vol_ma_ratio","momentum20","atr","dd_from_high"
]

# ====== DATA ======
def load_market_index(start, end):
    idx = ["SPY","QQQ","^VIX"]
    data = yf.download(idx, start=start, end=end, auto_adjust=True, group_by='ticker', threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs("Close", axis=1, level=1)
    else:
        close = data["Close"].to_frame()
    close = close.rename(columns={"SPY":"spy","QQQ":"qqq","^VIX":"vix"})
    rets = close.pct_change().rename(columns={"spy":"spy_ret","qqq":"qqq_ret","vix":"vix_ret"}).reset_index()
    rets = rets.rename(columns={"Date":"date"})
    rets["date"] = pd.to_datetime(rets["date"])
    return rets

def load_all_data_fast(tickers, start, end):
    print(f"Downloading {len(tickers)} tickers in batch...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True,
                       group_by='ticker', threads=True, progress=False)
    market = load_market_index(start, end)
    rows = []
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if t not in data.columns.levels[0]:
                    print(f"⚠ Missing: {t}"); continue
                df = data[t].copy().reset_index()
                df.columns = [c.lower() for c in df.columns]
            else:
                df = data.copy().reset_index()
                df.columns = [c.lower() for c in df.columns]
            need = {"date","open","high","low","close","volume"}
            if not need.issubset(df.columns):
                print(f"⚠ Columns missing: {t}"); continue
            feat = make_features(df[list(need)], t).dropna()
            merged = pd.merge(feat, market, on="date", how="left")
            rows.append(merged)
        except Exception as e:
            print(f"⚠ {t}: {e}")
    if not rows: raise RuntimeError("No data.")
    all_df = pd.concat(rows, ignore_index=True).sort_values(["date","ticker"]).reset_index(drop=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    return all_df

# ====== LABELS ======
def add_future_labels_inplace(df: pd.DataFrame):
    for n in [1,3,5]:
        df[f"ret_next{n}"] = df.groupby("ticker")["close"].shift(-n)/df["close"] - 1
        df[f"label{n}"] = (df[f"ret_next{n}"] > 0).astype(int)
    return df

# ====== MODEL ======
def _lgb_params(device_try_gpu=True, **over):
    params = {
        "objective":"binary","metric":"auc","boosting_type":"gbdt",
        "random_state":CFG["RANDOM_SEED"],"verbosity":-1,
        "learning_rate":CFG["LGBM"]["learning_rate"],
        "num_leaves":CFG["LGBM"]["num_leaves"],
        "n_estimators":CFG["LGBM"]["n_estimators"],
        "max_depth":CFG["LGBM"]["max_depth"],
    }
    params.update(over)
    if device_try_gpu:
        try:
            params["device_type"] = "gpu"
            _ = lgb.train(params, lgb.Dataset(np.zeros((5,3)), label=np.zeros(5)))
        except Exception:
            params.pop("device_type", None)
    return params

def train_models(train_df: pd.DataFrame):
    params = _lgb_params(device_try_gpu=True)
    models = {}
    for n in [1,3,5]:
        sub = train_df.dropna(subset=[f"label{n}"])
        if sub.empty: continue
        X, y = sub[FEATS], sub[f"label{n}"]
        models[n] = lgb.train(params, lgb.Dataset(X, label=y))
    return models

def predict_ensemble(models: dict, df: pd.DataFrame):
    X = df[FEATS]
    preds, ws = [], []
    for n, w in zip([1,3,5],[0.5,0.3,0.2]):
        if n in models:
            preds.append(models[n].predict(X))
            ws.append(w)
    if not preds: return np.zeros(len(df))
    return np.average(np.vstack(preds), axis=0, weights=ws)

# ====== UTILS ======
def softmax(x, temp=1.0):
    x = np.array(x, dtype=float)
    x = x - np.max(x)
    z = np.exp(x/float(temp))
    p = z / (z.sum() + 1e-12)
    return p

def load_sector_map(path):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        m = dict(zip(df["ticker"], df["sector"]))
        return m
    return {}

def apply_sector_cap(rows, max_per_sector, sector_map):
    if not max_per_sector or max_per_sector is None:
        return rows
    used = {}
    kept = []
    for r in rows:
        sec = sector_map.get(r["ticker"], "UNKNOWN")
        used[sec] = used.get(sec, 0) + 1
        if used[sec] <= max_per_sector:
            kept.append(r)
    return kept

def post_discord(webhook, title, lines):
    if not webhook: 
        print("(Discord webhook not set; skip)"); return
    content = f"📊 **{title}**\n" + "\n".join(lines)
    r = requests.post(webhook, json={"content": content})
    print("Discord:", "OK" if r.status_code in (200,204) else f"NG {r.status_code}: {r.text}")

# ====== CORE RUNNER ======
def run_production_day():
    start, end = CFG["START"], CFG["END"]
    # 1) データ
    all_df = load_all_data_fast(ALL_TICKERS, start, end)
    last_day = all_df["date"].max()
    train_start = last_day - pd.DateOffset(months=CFG["TRAIN_MONTHS"])
    tr = all_df[(all_df["date"]>=train_start) & (all_df["date"]<=last_day)].copy()

    # 2) ラベル（未来遮断）
    add_future_labels_inplace(tr)

    # 3) 学習
    models = train_models(tr)

    # 4) 予測（最新日＝翌営業日用）
    today_df = all_df[all_df["date"]==last_day].copy()
    today_df["prob"] = predict_ensemble(models, today_df)

    # 5) スコア要約
    mean_prob = float(today_df["prob"].mean())
    top = today_df.sort_values("prob", ascending=False).head(CFG["TOP_K"]).copy()

    # 6) 参戦判定（休む/戦う）
    go_trade = (mean_prob >= CFG["PROB_TH"])
    if not go_trade:
        msg = f"平均prob={mean_prob:.3f} < TH={CFG['PROB_TH']:.2f} → 本日ノーポジ"
        print(msg)

    # 7) ソフトマックス配分
    pick_rows = [{"ticker": r.ticker, "prob": float(r.prob), "close": float(r.close)} 
                 for _, r in top.iterrows()]
    # セクター制限
    sector_map = load_sector_map(CFG["SECTOR_MAP_CSV"])
    pick_rows = apply_sector_cap(pick_rows, CFG["MAX_PER_SECTOR"], sector_map)

    probs = [r["prob"] for r in pick_rows]
    if len(probs)==0 or not go_trade:
        weights = np.zeros(len(pick_rows))
    else:
        weights = softmax(probs, temp=CFG["SOFTMAX_TEMP"])
        # 極小ウェイト切り捨て・正規化
        weights = np.where(weights >= CFG["MIN_WEIGHT_CUT"], weights, 0.0)
        s = weights.sum()
        weights = weights / s if s>0 else weights

    # 8) 実際の資金配分（上限適用）
    allocs = []
    gross = 0.0
    for w, r in zip(weights, pick_rows):
        w = min(w, CFG["POSITION_CAP_PCT"])
        allocs.append({"ticker": r["ticker"], "weight": float(w), "prob": r["prob"], "close": r["close"]})
        gross += w
    # 総和が MAX_GROSS_EXPOSURE を超えるなら縮小
    if gross > 0 and gross > CFG["MAX_GROSS_EXPOSURE"]:
        scale = CFG["MAX_GROSS_EXPOSURE"] / gross
        for a in allocs: a["weight"] *= scale
        gross = CFG["MAX_GROSS_EXPOSURE"]

    # 9) 発注リスト作成（理論口数）
    orders = []
    for a in allocs:
        notional = CFG["DAILY_CAPITAL"] * a["weight"]
        px = a["close"] * (1 + CFG["SLIPPAGE"])  # 成行前提の控えめ約定想定
        qty = math.floor(notional / px) if px>0 else 0
        if qty <= 0: 
            continue
        orders.append({
            "ticker": a["ticker"],
            "target_weight": round(a["weight"], 4),
            "prob": round(a["prob"], 4),
            "price_assumed": round(px, 2),
            "qty": int(qty),
            "notional": round(qty * px, 2),
        })

    # 10) 保存 & 通知
    stamp = str(last_day.date())
    out_dir = CFG["OUT_DIR"]
    # Signals CSV
    if CFG["SAVE_SIGNALS_CSV"]:
        sig_cols = ["ticker","prob","close","rsi","ma_gap","volatility"]
        sig_df = today_df.sort_values("prob", ascending=False)[sig_cols].head(20).copy()
        sig_df["prob"] = sig_df["prob"].round(4)
        sig_path = os.path.join(out_dir, f"signals_{stamp}.csv")
        sig_df.to_csv(sig_path, index=False)
        print("Saved:", sig_path)
    # Orders CSV
    if CFG["SAVE_ORDERS_CSV"]:
        ord_df = pd.DataFrame(orders)
        ord_path = os.path.join(out_dir, f"orders_{stamp}.csv")
        ord_df.to_csv(ord_path, index=False)
        print("Saved:", ord_path)
    # Summary JSON
    summary = {
        "date": stamp,
        "go_trade": go_trade,
        "mean_prob": round(mean_prob, 4),
        "top_k": CFG["TOP_K"],
        "gross_exposure": round(float(sum(a["target_weight"] for a in orders)), 4) if orders else 0.0,
        "daily_capital": CFG["DAILY_CAPITAL"],
        "orders_count": len(orders),
    }
    if CFG["SAVE_SUMMARY_JSON"]:
        js_path = os.path.join(out_dir, f"summary_{stamp}.json")
        with open(js_path, "w") as f: json.dump(summary, f, indent=2)
        print("Saved:", js_path)

    # Discord
    lines = [
        f"日時(JST): {datetime.datetime.now(TZ).strftime('%Y-%m-%d %H:%M')}",
        f"データ最終日: {stamp}",
        f"学習窓: {CFG['TRAIN_MONTHS']}ヶ月 | 参戦TH: {CFG['PROB_TH']:.2f} | 平均prob: {mean_prob:.3f}",
        f"TOP_K: {CFG['TOP_K']} | 温度T: {CFG['SOFTMAX_TEMP']:.2f} | セク上限: {CFG['MAX_PER_SECTOR']}",
        "—— 発注候補 ——" if orders else "—— 本日ノーポジ ——",
    ]
    for o in orders[:10]:
        lines.append(f"{o['ticker']} | w={o['target_weight']:.3f} | prob={o['prob']:.3f} | qty={o['qty']} @ ${o['price_assumed']:.2f}")
    lines.append(f"総エクスポージャー(理論): {summary['gross_exposure']:.3f}")
    if CFG["SAVE_SIGNALS_CSV"]: lines.append(f"Signals: {sig_path}")
    if CFG["SAVE_ORDERS_CSV"]:  lines.append(f"Orders:  {ord_path}")
    if CFG["SAVE_SUMMARY_JSON"]:lines.append(f"Summary: {js_path}")
    post_discord(CFG["DISCORD_WEBHOOK"], "AI株式シミュレーター 実運用レポート", lines)

    return {"orders": orders, "summary": summary}

# ====== RUN ======
res = run_production_day()
res



