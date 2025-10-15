#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parameta_offline.py
オフライン（CSVキャッシュのみ）で動くパラメータ探索機。
- yfinance等は使わない
- price_cache/*.csv から読み込み
- Walk-Forwardで評価
- OptunaでトップK/配分温度/しきい値/収益クリップ等を探索
"""

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import optuna

# =========================
# 既定パラメータ
# =========================
RANDOM_SEED = 42
TRAIN_MONTHS = 6
TEST_MONTHS  = 1
COMMISSION   = 0.0005

# モデル固定（ユーザー指示：年率32%の時の設計に準拠する構造）
LGB_BASE_PARAMS = {
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

# =========================
# ユニバース（例）— 実際には price_cache にある銘柄だけを使ってOK
# =========================
DEFAULT_TICKERS = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AMD","INTC","AVGO",
    "ADBE","CRM","NOW","PANW","DDOG","SNOW","CRWD","ZS","NET","MDB",
    "COST","WMT","HD","LOW","NKE","MCD","SBUX","KO","PEP","PG",
    "BKNG","ABNB","UBER","JPM","BAC","GS","MS","BLK","SPGI","V",
    "MA","AXP","XOM","CVX","COP","PSX","MPC","OXY","SLB","EOG",
    "DVN","FCX","NEM","CAT","DE","HON","GE","BA","UPS","FDX",
    "LMT","RTX","NOC","GD","DIS","NFLX","PYPL","SHOP","SQ","PLTR",
    "ORCL","IBM","QCOM","TXN","MU","CSCO","TMO","DHR","LLY","MRK",
    "PFE","BMY","JNJ","UNH","CI","HUM","CVS","WMT","HD","LOW" # 重複OKでも読み込みは一意化
]

# =========================
# 便利関数
# =========================
def rsi(series: pd.Series, n=14):
    diff = series.diff()
    up = diff.clip(lower=0)
    dn = -diff.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_dn + 1e-12)
    return 100 - (100 / (1 + rs))

def ensure_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).tz_localize(None)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    CSV取り込み後の列を標準化：
    - Date/date → 'date'
    - open/high/low/close/volume を適切に推定
    """
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    date_col = pick("date","datetime","timestamp")
    if date_col is None:
        # 先頭列を日付とみなす最後の手段
        date_col = df.columns[0]
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")

    for std, candidates in {
        "Open":   ("open","o","opn"),
        "High":   ("high","h","hi"),
        "Low":    ("low","l","lo"),
        "Close":  ("close","adj close","adj_close","c"),
        "Volume": ("volume","vol","v")
    }.items():
        src = pick(*candidates)
        if src is None:
            # 最低限 Close が無いと後工程が成り立たない
            if std == "Close":
                raise ValueError("missing column: Close")
            else:
                out[std] = np.nan
        else:
            out[std] = pd.to_numeric(df[src], errors="coerce")

    out = out.dropna(subset=["date","Close"]).sort_values("date")
    return out.reset_index(drop=True)

def load_csv_ticker(cache_dir: str, ticker: str) -> pd.DataFrame:
    """
    price_cache/ticker.csv を読み込み
    """
    # 記号はファイル名に使いにくいので簡易変換
    fname = ticker.replace("^", "").replace("/", "_")
    path = os.path.join(cache_dir, f"{fname}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing cache: {path}")
    raw = pd.read_csv(path)
    df = normalize_columns(raw)
    df["ticker"] = ticker
    return df

def load_universe(cache_dir: str, tickers: List[str], start: str, end: str, verbose=True) -> pd.DataFrame:
    start_dt = ensure_date(start); end_dt = ensure_date(end)
    out = []
    seen = set()
    for t in tickers:
        if t in seen: 
            continue
        try:
            df = load_csv_ticker(cache_dir, t)
            df = df[(df["date"]>=start_dt) & (df["date"]<=end_dt)].copy()
            if len(df) == 0:
                if verbose: print(f"[skip-empty] {t}")
                continue
            out.append(df)
            seen.add(t)
            if verbose: print(f"[ok] {t}: {len(df)} rows")
        except Exception as e:
            if verbose: print(f"[fail] {t}: {e}")
    if not out:
        raise RuntimeError("No price data fetched from cache. Put CSVs into price_cache/")
    all_df = pd.concat(out, ignore_index=True).sort_values(["date","ticker"]).reset_index(drop=True)
    return all_df

def load_market_index(cache_dir: str, start: str, end: str) -> pd.DataFrame:
    """
    任意（SPY/QQQ/VIX）が price_cache に在れば使う。無ければ空を返す。
    """
    idx_map = {"SPY":"spy_ret","QQQ":"qqq_ret","VIX":"vix_ret"}
    out = []
    for symbol, colname in idx_map.items():
        try:
            df = load_csv_ticker(cache_dir, symbol)
            df = df[(df["date"]>=ensure_date(start)) & (df["date"]<=ensure_date(end))][["date","Close"]].copy()
            df = df.rename(columns={"Close": colname})
            df[colname] = df[colname].pct_change()
            out.append(df)
        except Exception:
            pass
    if not out: 
        return pd.DataFrame()
    m = out[0]
    for k in out[1:]:
        m = pd.merge(m, k, on="date", how="outer")
    return m.sort_values("date").reset_index(drop=True)

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c, v = df["Close"].astype(float), df["Volume"].astype(float)
    f = pd.DataFrame(index=df.index)
    f["sma_s"] = c.rolling(10).mean()
    f["sma_m"] = c.rolling(30).mean()
    f["sma_l"] = c.rolling(90).mean()
    f["rsi"]   = rsi(c, 14)
    f["ret1"]  = c.pct_change(1)
    f["ret5"]  = c.pct_change(5)
    f["ret10"] = c.pct_change(10)
    f["vol_chg"] = v.pct_change()
    f["ma_gap"] = (c - f["sma_m"]) / (f["sma_m"] + 1e-12)
    f["volatility"] = c.pct_change().rolling(20).std()
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_upper"] = (c - (ma20 + 2*sd20)) / (2*sd20 + 1e-12)
    f["bb_lower"] = (c - (ma20 - 2*sd20)) / (2*sd20 + 1e-12)
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    f["macd"] = macd
    f["macd_signal"] = macd - macd.ewm(span=9).mean()
    f["vol_ma_ratio"] = v / (v.rolling(20).mean() + 1e-12)
    f["momentum20"] = c / c.shift(20) - 1
    f["atr"] = (df["High"] - df["Low"]).rolling(14).mean()
    f["dd_from_high"] = c / c.rolling(20).max() - 1
    return f

FEATS = [
    "sma_s","sma_m","sma_l","rsi","ret1","ret5","ret10","vol_chg",
    "ma_gap","volatility","bb_upper","bb_lower","macd","macd_signal",
    "spy_ret","qqq_ret","vix_ret","vol_ma_ratio","momentum20","atr","dd_from_high"
]

def add_future_labels_inplace(df: pd.DataFrame):
    for n in [1,3,5]:
        df[f"ret_next{n}"] = df.groupby("ticker")["Close"].shift(-n)/df["Close"] - 1
        df[f"label{n}"] = (df[f"ret_next{n}"]>0).astype(int)

# =========================
# 学習・予測
# =========================
def train_models(train_df: pd.DataFrame) -> Dict[str, lgb.Booster]:
    def _train(target):
        sub = train_df.dropna(subset=[target])
        if sub.empty: return None
        X, y = sub[FEATS], sub[target].astype(int)
        dtrain = lgb.Dataset(X, label=y)
        return lgb.train(LGB_BASE_PARAMS, dtrain)
    return {
        "m1": _train("label1"),
        "m3": _train("label3"),
        "m5": _train("label5"),
    }

def predict_models(models: Dict[str,lgb.Booster], X: pd.DataFrame) -> Dict[str,np.ndarray]:
    def _pred(m):
        if m is None: return np.zeros(len(X))
        return m.predict(X)
    return {
        "p1": _pred(models["m1"]),
        "p3": _pred(models["m3"]),
        "p5": _pred(models["m5"]),
    }

def estimate_mu_by_fold(train_df: pd.DataFrame) -> Dict[str,float]:
    mu = {}
    for k in ["1","3","5"]:
        lab = train_df[f"label{k}"]; ret = train_df[f"ret_next{k}"]
        m = (lab==1) & (~ret.isna())
        mu[k] = float(ret[m].mean()) if m.any() else 0.0
        if not np.isfinite(mu[k]): mu[k] = 0.0
    return mu

def softmax_alloc(p, temp=0.3):
    x = np.clip(np.asarray(p, dtype=float), 1e-9, 1-1e-9) / max(1e-6, temp)
    w = np.exp(x - x.max()); w /= w.sum()
    return w

# =========================
# Walk-Forward + Backtest（パラメータで挙動変化）
# =========================
@dataclass
class WFParams:
    per_pos_frac: float = 0.10
    top_k: int = 3
    softmax_temp: float = 0.35
    fixed_score_quantile: float = 0.55
    ret_clip_low: float = -0.08
    ret_clip_high: float = 0.30

def backtest_pxmu_daily(df: pd.DataFrame, top_k: int, per_pos_frac: float, softmax_temp: float) -> Tuple[pd.Series, Dict]:
    g = df.sort_values(["date","score"], ascending=[True,False]).groupby("date")
    cap = 1_000_000.0
    eq_vals, dates = [], []
    for d, sub in g:
        sub = sub.dropna(subset=["ret_chosen","score"])
        if sub.empty: continue
        ranked = list(sub["ticker"].values)
        ranked = [t for _, t in sorted(zip(sub["score"].values, ranked), reverse=True)]
        picks = ranked[:top_k]
        use = sub[sub["ticker"].isin(picks)].copy()
        if use.empty: continue
        w = softmax_alloc(use["score"].values, temp=softmax_temp)
        r = np.clip(use["ret_chosen"].values, -0.99, 10.0)  # まず安全に
        r = np.clip(r, RET_CLIP_LOW, RET_CLIP_HIGH) - 2*COMMISSION
        day_ret = np.sum(w * r) * per_pos_frac
        cap *= (1 + day_ret)
        eq_vals.append(cap); dates.append(d)
    if not eq_vals: 
        return pd.Series(dtype=float), {"total_return":np.nan,"sharpe":np.nan,"max_dd":np.nan}
    eq = pd.Series(eq_vals, index=pd.to_datetime(dates))
    dr = eq.pct_change().dropna()
    sharpe = (dr.mean()/(dr.std()+1e-12))*np.sqrt(252) if len(dr)>1 else np.nan
    max_dd = ((eq.cummax()-eq)/eq.cummax()).max() if len(eq)>1 else np.nan
    return eq, {"total_return": eq.iloc[-1]/eq.iloc[0]-1, "sharpe":sharpe, "max_dd":max_dd}

def walk_forward_eval(all_df: pd.DataFrame, wfp: WFParams) -> Tuple[pd.Series, Dict]:
    df = all_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    months = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")[TRAIN_MONTHS:]
    cap = 1_000_000.0
    equity_all = []
    for ms in months:
        tr_start = ms - pd.DateOffset(months=TRAIN_MONTHS)
        tr_end   = ms - pd.DateOffset(days=1)
        te_start = ms
        te_end   = ms + pd.DateOffset(months=TEST_MONTHS) - pd.DateOffset(days=1)

        tr = df[(df["date"]>=tr_start)&(df["date"]<=tr_end)].copy()
        te = df[(df["date"]>=te_start)&(df["date"]<=te_end)].copy()
        if len(tr)<300 or len(te)<20: 
            continue

        add_future_labels_inplace(tr); add_future_labels_inplace(te)

        models = train_models(tr)
        mu = estimate_mu_by_fold(tr)

        # validation: train末30日
        val_end = tr["date"].max()
        val_start = val_end - pd.DateOffset(days=30)
        val = tr[tr["date"].between(val_start, val_end)].dropna(subset=FEATS).copy()

        # 予測
        p_val = predict_models(models, val[FEATS])
        p_te  = predict_models(models, te[FEATS].dropna())

        score_val_best = np.vstack([p_val["p1"]*mu["1"], p_val["p3"]*mu["3"], p_val["p5"]*mu["5"]]).T
        thr = np.quantile(score_val_best.max(axis=1), wfp.fixed_score_quantile) if len(score_val_best)>0 else 1e9

        # テスト期間のベストhorizonとスコア
        te_use = te.dropna(subset=FEATS+["ret_next1","ret_next3","ret_next5"]).copy()
        p_use  = predict_models(models, te_use[FEATS])
        score_te = np.vstack([p_use["p1"]*mu["1"], p_use["p3"]*mu["3"], p_use["p5"]*mu["5"]]).T
        best_idx = np.argmax(score_te, axis=1) if len(te_use)>0 else np.array([])
        best_score = score_te.max(axis=1) if len(te_use)>0 else np.array([])

        if len(te_use)>0:
            te_use["score"] = best_score
            ret1, ret3, ret5 = te_use["ret_next1"].values, te_use["ret_next3"].values, te_use["ret_next5"].values
            te_use["ret_chosen"] = np.where(best_idx==0, ret1, np.where(best_idx==1, ret3, ret5))
            te_use = te_use[te_use["score"]>=thr-1e-12][["date","ticker","score","ret_chosen"]]

            eq, _ = backtest_pxmu_daily(te_use, top_k=wfp.top_k, per_pos_frac=wfp.per_pos_frac, softmax_temp=wfp.softmax_temp)
            if len(eq)>1:
                equity_all.append(eq)

    if not equity_all:
        return pd.Series(dtype=float), {"total_return":np.nan,"sharpe":np.nan,"max_dd":np.nan}
    eq_total = pd.concat(equity_all).sort_index()
    dr = eq_total.pct_change().dropna()
    metrics = {
        "total_return": eq_total.iloc[-1]/eq_total.iloc[0]-1,
        "sharpe": (dr.mean()/(dr.std()+1e-12))*np.sqrt(252) if len(dr)>1 else np.nan,
        "max_dd": ((eq_total.cummax()-eq_total)/eq_total.cummax()).max() if len(eq_total)>1 else np.nan
    }
    return eq_total, metrics

# =========================
# Optuna 目的関数
# =========================
def objective(all_df: pd.DataFrame, trial: optuna.Trial):
    wfp = WFParams(
        per_pos_frac    = trial.suggest_float("per_pos_frac", 0.05, 0.25),
        top_k           = trial.suggest_int("top_k", 1, 5),
        softmax_temp    = trial.suggest_float("softmax_temp", 0.05, 0.6),
        fixed_score_quantile = trial.suggest_float("fixed_score_quantile", 0.45, 0.65),
        ret_clip_low    = trial.suggest_float("ret_clip_low", -0.20, -0.04),
        ret_clip_high   = trial.suggest_float("ret_clip_high", 0.12, 0.50),
    )
    global RET_CLIP_LOW, RET_CLIP_HIGH
    RET_CLIP_LOW, RET_CLIP_HIGH = wfp.ret_clip_low, wfp.ret_clip_high

    eq, m = walk_forward_eval(all_df, wfp)
    # 目的：年率（CAGR proxy）とドローダウンの折衷
    if len(eq) < 5 or not np.isfinite(m["total_return"]):
        return -9999.0
    cagr_like = m["total_return"]
    maxdd = m["max_dd"] if np.isfinite(m["max_dd"]) else 1.0
    score = cagr_like - 0.5*maxdd
    return float(score)

# =========================
# メイン
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="price_cache", help="CSVキャッシュフォルダ")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end",   default="2025-10-15")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--jobs",   type=int, default=1)
    ap.add_argument("--tickers_file", default="", help="銘柄リストを1行1銘柄で格納したtxt（任意）")
    args = ap.parse_args()

    # 銘柄
    if args.tickers_file and os.path.exists(args.tickers_file):
        with open(args.tickers_file, "r") as f:
            tickers = [ln.strip() for ln in f if ln.strip()]
    else:
        tickers = DEFAULT_TICKERS

    print(f"[run] tickers={len(tickers)}, START={args.start}, END={args.end}, trials={args.trials}, jobs={args.jobs}")
    # 価格読み込み
    px = load_universe(args.cache, tickers, args.start, args.end, verbose=True)
    # 特徴
    feats = []
    for t, g in px.groupby("ticker", sort=False):
        f = make_features(g)
        f["date"] = g["date"].values
        f["ticker"] = t
        feats.append(f.reset_index(drop=True))
    F = pd.concat(feats, ignore_index=True)

    ALL = pd.merge(px[["date","ticker","Open","High","Low","Close","Volume"]],
                   F, on=["date","ticker"], how="left")

    # 市場指標（任意）
    mkt = load_market_index(args.cache, args.start, args.end)
    if not mkt.empty:
        ALL = pd.merge(ALL, mkt, on="date", how="left")
    else:
        # 指標が無ければ0で埋める（特徴の次元は維持）
        for col in ["spy_ret","qqq_ret","vix_ret"]:
            ALL[col] = 0.0

    add_future_labels_inplace(ALL)

    # 欠損落とし＆整列
    ALL = ALL.dropna(subset=["Close"]).sort_values(["date","ticker"]).reset_index(drop=True)

    # Optuna
    def _obj(trial):
        return objective(ALL, trial)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(_obj, n_trials=args.trials, n_jobs=args.jobs, show_progress_bar=True)

    best = study.best_trial
    print("\n[best] value=", best.value)
    print("[best] params=", json.dumps(best.params, indent=2, ensure_ascii=False))

    # 保存
    pd.DataFrame([{"trial":t.number, **t.params, "value":t.value} for t in study.trials]).to_csv("optuna_trials_summary.csv", index=False)
    with open("best_params_offline.json","w") as f:
        json.dump(best.params, f, indent=2)
    print("[save] optuna_trials_summary.csv, best_params_offline.json")

if __name__ == "__main__":
    # グローバルに使うクリップ値を初期化（objectiveで上書き）
    RET_CLIP_LOW, RET_CLIP_HIGH = -0.08, 0.30
    main()
