# -*- coding: utf-8 -*-
"""
決定版：AI株式シミュレーター（6m→1m｜Aggressive｜Discord通知付）
- LightGBM（GPU自動使用）
- p×μ法（1/3/5日保有を日次最適選択）
- 確率校正（Isotonic/Platt）
- 固定分位しきい値（攻め型）
- 温度ソフトマックス集中配分
- セクター上限解除
- Discord通知（前日比・収益額・累計）
"""

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import warnings, os, requests
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# =========================
# 設定
# =========================
START = "2015-01-01"
END   = None
RANDOM_SEED = 42

TRAIN_MONTHS = 6
TEST_MONTHS  = 1

TOP_K = 3
INIT_CAPITAL = 1_000_000
PER_POS_FRAC = 0.20
COMMISSION   = 0.0005

USE_CALIBRATION  = True
CALIB_METHOD     = "isotonic"   # or "platt"

USE_DYNAMIC_TH   = False
FIXED_SCORE_QUANTILE = 0.50

SOFTMAX_TEMP     = 0.15
MIN_WEIGHT_CUT   = 0.00

RET_CLIP_LOW, RET_CLIP_HIGH = -0.08, 0.30

MAX_PER_SECTOR = None
SKIP_SECTOR_FOR_UNKNOWN = True

# =========================
# LightGBM パラメータ
# =========================
def _lgb_params(device_try_gpu=True):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "random_state": RANDOM_SEED,
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_estimators": 400,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 3,
        "min_child_samples": 40,
        "lambda_l2": 5.0,
    }
    if device_try_gpu:
        try:
            params["device_type"] = "gpu"
            _ = lgb.train(params, lgb.Dataset(np.zeros((5,3)), label=np.zeros(5)))
        except Exception:
            params.pop("device_type", None)
    return params

# =========================
# ユニバース（代表100銘柄）※SQ除外（yfinanceのTZエラー回避）
# =========================
ALL_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AMD","NFLX","ADBE",
    "CRM","INTC","IBM","ORCL","QCOM","AVGO","CSCO","TXN","MU","SHOP",
    "SNOW","PANW","TEAM","DDOG","PLTR","UBER","ABNB","PYPL","NOW","ZM",
    "CRWD","MDB","RBLX","NET","ZS","COST","WMT","HD","LOW","TGT",
    "MCD","SBUX","NKE","KO","PEP","PG","PM","DIS","BKNG","F",
    "GM","CAT","BA","GE","DE","UPS","FDX","HON","MMM","LMT",
    "RTX","NOC","GD","XOM","CVX","COP","PSX","MPC","OXY","SLB",
    "EOG","DVN","APA","FCX","NEM","JPM","BAC","C","MS","GS",
    "BLK","SCHW","SPGI","ICE","V","MA","AXP","JNJ","PFE","MRK",
    "BMY","LLY","UNH","CI","HUM","CVS","TMO","DHR","MDT"
]

# =========================
# テクニカル指標
# =========================
def rsi(series: pd.Series, n=14):
    diff = series.diff()
    up = diff.clip(lower=0)
    dn = -diff.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_dn + 1e-12)
    return 100 - (100 / (1 + rs))

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
    return df.dropna().reset_index(drop=True)

FEATS = [
    "sma_s","sma_m","sma_l","rsi","ret1","ret5","ret10","vol_chg",
    "ma_gap","volatility","bb_upper","bb_lower","macd","macd_signal",
    "spy_ret","qqq_ret","vix_ret","vol_ma_ratio","momentum20","atr","dd_from_high"
]

# =========================
# 市場データ
# =========================
def load_market_index():
    idx_list = ["SPY","QQQ","^VIX"]
    data = yf.download(idx_list, start=START, end=END, auto_adjust=True,
                       group_by='ticker', threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs('Close', axis=1, level=1)
    else:
        close = data["Close"].to_frame()
    close = close.rename(columns={"SPY":"spy","QQQ":"qqq","^VIX":"vix"})
    rets = close.pct_change().rename(columns={"spy":"spy_ret","qqq":"qqq_ret","vix":"vix_ret"}).reset_index()
    rets = rets.rename(columns={"Date":"date"})
    rets["date"] = pd.to_datetime(rets["date"])
    return rets

def load_all_data_fast(tickers):
    print(f"Downloading {len(tickers)} tickers in batch...")
    data = yf.download(tickers, start=START, end=END, auto_adjust=True,
                       group_by='ticker', threads=True, progress=False)
    market = load_market_index()
    out = []
    is_multi = isinstance(data.columns, pd.MultiIndex)
    for t in tickers:
        try:
            if is_multi:
                if t not in data.columns.get_level_values(0):
                    print(f"⚠ データ無し: {t}")
                    continue
                df = data[t].copy().reset_index()
                df.columns = [c.lower() for c in df.columns]
            else:
                df = data.copy().reset_index()
                df.columns = [c.lower() for c in df.columns]

            need = {"date","open","high","low","close","volume"}
            if not need.issubset(df.columns):
                print(f"⚠ 欠損列: {t}")
                continue

            feat = make_features(df[list(need)], t)
            merged = pd.merge(feat, market, on="date", how="left")
            out.append(merged)
        except Exception as e:
            print(f"⚠ {t}: {e}")
            continue
    if not out:
        raise RuntimeError("No price data fetched. Check tickers or network.")
    all_df = pd.concat(out, ignore_index=True).sort_values(["date","ticker"]).reset_index(drop=True)
    return all_df

# =========================
# 学習・推論・評価
# =========================
def add_future_labels_inplace(df: pd.DataFrame):
    for n in [1,3,5]:
        df[f"ret_next{n}"] = df.groupby("ticker")["close"].shift(-n) / df["close"] - 1
        df[f"label{n}"] = (df[f"ret_next{n}"] > 0).astype(int)
    return df

def train_one_model(train_df, target_col, params):
    df = train_df.dropna(subset=[target_col])
    if df.empty: return None
    X, y = df[FEATS], df[target_col].astype(int)
    return lgb.train(params, lgb.Dataset(X, label=y))

def train_models(train_df):
    params = _lgb_params(device_try_gpu=True)
    return {"m1": train_one_model(train_df, "label1", params),
            "m3": train_one_model(train_df, "label3", params),
            "m5": train_one_model(train_df, "label5", params)}

def predict_models(models, df):
    X = df[FEATS]
    return {
        "p1": models["m1"].predict(X) if models["m1"] else np.zeros(len(df)),
        "p3": models["m3"].predict(X) if models["m3"] else np.zeros(len(df)),
        "p5": models["m5"].predict(X) if models["m5"] else np.zeros(len(df)),
    }

def fit_calibrator(y_true, p_raw, method="isotonic"):
    y_true, p_raw = np.asarray(y_true), np.asarray(p_raw)
    m = ~np.isnan(y_true) & ~np.isnan(p_raw)
    y_true, p_raw = y_true[m], p_raw[m]
    if len(y_true) < 20: return None
    if method=="isotonic":
        iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(p_raw, y_true); return ("isotonic", iso)
    lr = LogisticRegression(max_iter=1000); lr.fit(p_raw.reshape(-1,1), y_true.astype(int)); return ("platt", lr)

def apply_calibrator(calib, p_raw):
    if calib is None: return p_raw
    kind, model = calib
    p_raw = np.asarray(p_raw)
    return model.transform(p_raw) if kind=="isotonic" else model.predict_proba(p_raw.reshape(-1,1))[:,1]

def estimate_mu_by_fold(train_df):
    mu = {}
    for k in ["1","3","5"]:
        lab, ret = train_df[f"label{k}"], train_df[f"ret_next{k}"]
        m = (lab == 1) & (~ret.isna())
        mu[k] = float(ret[m].mean()) if m.any() else 0.0
        if not np.isfinite(mu[k]): mu[k] = 0.0
    return mu

def softmax_alloc(p, temp=SOFTMAX_TEMP, min_w=MIN_WEIGHT_CUT):
    x = np.clip(np.asarray(p, dtype=float), 1e-9, 1-1e-9) / max(1e-6, temp)
    w = np.exp(x - x.max()); w /= w.sum()
    return w

# =========================
# バックテスト & ウォークフォワード
# =========================
def backtest_pxmu_daily(df, top_k=TOP_K, initial_capital=INIT_CAPITAL):
    g = df.sort_values(["date","score"], ascending=[True,False]).groupby("date")
    eq_vals, dates, cap = [], [], float(initial_capital)
    for d, sub in g:
        sub = sub.dropna(subset=["ret_chosen","score"])
        if sub.empty: continue
        ranked = sub.sort_values("score", ascending=False).head(top_k)
        w = softmax_alloc(ranked["score"].values, temp=SOFTMAX_TEMP)
        r = np.clip(ranked["ret_chosen"].values, RET_CLIP_LOW, RET_CLIP_HIGH) - 2*COMMISSION
        day_ret = np.sum(w * r) * PER_POS_FRAC
        cap *= (1 + day_ret)
        dates.append(d); eq_vals.append(cap)
    return pd.Series(eq_vals, index=pd.to_datetime(dates))

def walk_forward(all_df):
    df = all_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    months = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")[TRAIN_MONTHS:]
    equity_all, cap = [], INIT_CAPITAL
    for ms in months:
        train_start = ms - pd.DateOffset(months=TRAIN_MONTHS)
        train_end   = ms - pd.DateOffset(days=1)
        test_start  = ms
        test_end    = ms + pd.DateOffset(months=TEST_MONTHS) - pd.DateOffset(days=1)

        tr = df[(df["date"]>=train_start)&(df["date"]<=train_end)].copy()
        te = df[(df["date"]>=test_start)&(df["date"]<=test_end)].copy()
        if len(tr)<300 or len(te)<20: continue

        add_future_labels_inplace(tr)
        add_future_labels_inplace(te)

        models = train_models(tr)
        mu = estimate_mu_by_fold(tr)

        # validation = train末30日
        val_end = tr["date"].max()
        val_start = val_end - pd.DateOffset(days=30)
        val = tr[tr["date"].between(val_start, val_end)].copy()

        p_val_raw = predict_models(models, val)
        p_te_raw  = predict_models(models, te)

        if USE_CALIBRATION:
            calib1 = fit_calibrator(val["label1"], p_val_raw["p1"], method=CALIB_METHOD)
            calib3 = fit_calibrator(val["label3"], p_val_raw["p3"], method=CALIB_METHOD)
            calib5 = fit_calibrator(val["label5"], p_val_raw["p5"], method=CALIB_METHOD)
            p_te = {"p1": apply_calibrator(calib1, p_te_raw["p1"]),
                    "p3": apply_calibrator(calib3, p_te_raw["p3"]),
                    "p5": apply_calibrator(calib5, p_te_raw["p5"])}
        else:
            p_te = p_te_raw

        score_best = np.vstack([p_te["p1"]*mu["1"], p_te["p3"]*mu["3"], p_te["p5"]*mu["5"]]).T
        best_idx = np.argmax(score_best, axis=1)
        best_score = score_best.max(axis=1)

        te_use = te.copy()
        te_use["score"] = best_score
        te_use["ret_chosen"] = np.where(best_idx==0, te_use["ret_next1"],
                                 np.where(best_idx==1, te_use["ret_next3"], te_use["ret_next5"]))
        thr = np.quantile(best_score, FIXED_SCORE_QUANTILE)
        te_use = te_use[te_use["score"] >= (thr - 1e-12)]

        eq = backtest_pxmu_daily(te_use[["date","ticker","score","ret_chosen"]], top_k=TOP_K, initial_capital=cap)
        if len(eq.dropna())>1:
            equity_all.append(eq.dropna())
            cap = float(eq.iloc[-1])

    return (pd.concat(equity_all).sort_index() if equity_all else pd.Series(dtype=float))

# =========================
# ログ & 結果処理
# =========================
def summarize(eq: pd.Series):
    eq = eq.dropna()
    if len(eq)<2:
        return (np.nan, np.nan, np.nan)
    total = float(eq.iloc[-1]) / float(eq.iloc[0]) - 1
    dr = eq.pct_change().dropna()
    sharpe = (dr.mean()/(dr.std()+1e-12))*np.sqrt(252) if len(dr)>1 else np.nan
    maxdd = ((eq.cummax()-eq)/eq.cummax()).max() if len(eq)>1 else np.nan
    return total, sharpe, maxdd

def log_performance(eq: pd.Series):
    dr = eq.pct_change().dropna()
    if len(dr)==0:
        print("No daily returns"); return
    annual_ret = (1 + dr.mean())**252 - 1
    annual_vol = dr.std() * np.sqrt(252)
    print(f"平均日利: {dr.mean()*100:.3f}% | 年利: {annual_ret:.2%} | 年率ボラ: {annual_vol:.2%}")

# =========================
# Discord通知
# =========================
def notify_discord(result_text: str):
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        print("⚠️ No Discord webhook URL found. Skipping notification.")
        return
    payload = {
        "username": "AI Stock Bot",
        "embeds": [{
            "title": "📊 AIシミュレーター日次結果",
            "description": result_text,
            "color": 0x2ECC71
        }]
    }
    try:
        requests.post(url, json=payload, timeout=15)
        print("✅ Discord通知送信完了")
    except Exception as e:
        print(f"❌ Discord送信失敗: {e}")

# =========================
# 実行（Discord通知：前日比・収益額・累計）
# =========================
def main():
    all_df = load_all_data_fast(ALL_TICKERS)
    print(f"全データ件数: {len(all_df)}; 期間: {all_df['date'].min().date()} ～ {all_df['date'].max().date()}")
    eq = walk_forward(all_df)
    r, s, d = summarize(eq)
    print("\n=== Walk-Forward 総合結果（6m→1m｜Aggressive） ===")
    print(f"Return={r:.4f}, Sharpe={s:.4f}, MaxDD={d:.4f}")
    log_performance(eq)
    return eq, r, s, d

if __name__ == "__main__":
    eq, r, s, d = main()
    if eq is not None and len(eq) > 1:
        last_val = float(eq.iloc[-1])
        prev_val = float(eq.iloc[-2])
        daily_change_pct = (last_val / prev_val - 1.0) * 100.0
        daily_profit = last_val - prev_val
        total_profit = last_val - INIT_CAPITAL
        icon = "📈" if daily_profit >= 0 else "📉"

        result = (
            f"**Return:** {r:.4f} (累計)\n"
            f"**Sharpe:** {s:.4f}\n"
            f"**MaxDD:** {d:.4f}\n"
            f"**前日比:** {daily_change_pct:+.2f}% （{daily_profit:+,.0f}円） {icon}\n"
            f"**累計損益:** {total_profit:+,.0f}円"
        )
        notify_discord(result)
    else:
        print("⚠️ Equityデータが不十分のためDiscord通知スキップ")













