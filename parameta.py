# -*- coding: utf-8 -*-
"""
parameta.py - キャッシュ対応済みパラメータ探査機
LightGBM + Optuna による自動パラメータ探索
--------------------------------------------
- 価格データを price_cache/ に保存
- 再実行時はキャッシュを自動利用
- Yahooレート制限(429)に対して指数バックオフ付き再試行
"""

import os, time, random, json, optuna
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# ============ 設定 ============
START = "2015-01-01"
END = "2025-10-16"
CACHE_DIR = "price_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
N_TRIALS = 8
RANDOM_SEED = 42
THREADS = 4

# ============ 価格取得関数（キャッシュ＋バックオフ対応） ============
def fetch_cached_ticker(ticker, start, end, verbose=True):
    path = os.path.join(CACHE_DIR, f"{ticker}.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if len(df) > 0:
            if verbose: print(f"[cache] {ticker}: {len(df)} rows")
            return df

    for attempt in range(1, 6):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
            if df is None or df.empty:
                raise ValueError("empty")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.reset_index().rename(columns=str.lower)
            df["ticker"] = ticker
            df.to_parquet(path)
            if verbose: print(f"[save] {ticker} ({len(df)} rows) -> {path}")
            return df
        except Exception as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            if verbose:
                print(f"[retry {attempt}] {ticker}: {e} -> waiting {wait:.1f}s")
            time.sleep(wait)

    print(f"[fail] {ticker}: max retries exceeded")
    return None

# ============ 銘柄群ロード ============
def load_universe(tickers, start, end, verbose=True):
    dfs = []
    for t in tqdm(tickers, desc="[load] universe"):
        df = fetch_cached_ticker(t, start, end, verbose)
        if df is not None and not df.empty:
            dfs.append(df)
    if not dfs:
        raise RuntimeError("No price data fetched. Check tickers/start/end/network.")
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    return all_df.sort_values(["date", "ticker"]).reset_index(drop=True)

# ============ 特徴量生成 ============
def make_features(df):
    df = df.copy()
    df["ret1"] = df.groupby("ticker")["close"].pct_change()
    df["vol_chg"] = df.groupby("ticker")["volume"].pct_change()
    df["ma5"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(5).mean())
    df["ma20"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20).mean())
    df["mom5"] = df.groupby("ticker")["close"].transform(lambda x: x.pct_change(5))
    df["label"] = (df.groupby("ticker")["close"].shift(-5) / df["close"] - 1).clip(-0.2, 0.2)
    df = df.dropna().reset_index(drop=True)
    return df

# ============ 評価関数 ============
def walk_forward_eval(df, params, n_splits=5):
    df = df.sort_values("date")
    feats = ["ret1","vol_chg","ma5","ma20","mom5"]
    X, y = df[feats], (df["label"] > 0).astype(int)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    for i, (tr, te) in enumerate(tscv.split(X)):
        dtrain = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
        dtest = lgb.Dataset(X.iloc[te], label=y.iloc[te])
        model = lgb.train(params, dtrain, valid_sets=[dtest], verbose_eval=False)
        preds = model.predict(X.iloc[te])
        aucs.append(roc_auc_score(y.iloc[te], preds))
    return np.mean(aucs)

# ============ Optuna探索 ============
def run_optuna_search(tickers, start, end, n_trials_local, seed=42):
    all_df = load_universe(tickers, start, end)
    df = make_features(all_df)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "num_leaves": trial.suggest_int("num_leaves", 15, 80),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
            "seed": seed,
            "num_threads": THREADS
        }
        auc = walk_forward_eval(df, params)
        return auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    print(f"[optuna] optimizing ... trials: {n_trials_local}")
    study.optimize(objective, n_trials=n_trials_local, show_progress_bar=True)

    # 保存
    best_params = study.best_params
    json.dump(best_params, open("best_params.json", "w"), indent=2)
    study.trials_dataframe().to_csv("optuna_trials_summary.csv", index=False)
    print(f"[done] best params saved -> best_params.json (AUC={study.best_value:.4f})")

    return study

# ============ 実行部分 ============
if __name__ == "__main__":
    TICKERS = ["AAPL","MSFT","NVDA","META","TSLA","AMZN","GOOGL","JPM","KO","DIS"]
    print(f"[debug] START={START}, END={END}")
    study = run_optuna_search(TICKERS, START, END, n_trials_local=N_TRIALS, seed=RANDOM_SEED)
    print("✅ Parameter search completed successfully.")
