# -*- coding: utf-8 -*-
"""
generate_model_best.py
-----------------------------------------
parameta.py で探索済みパラメータ (best_params.json)
を使って LightGBM モデルを再学習・保存するスクリプト。
"""
import os, json, lightgbm as lgb, pandas as pd, numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

CACHE_DIR = "price_cache"
MODEL_PATH = "model_best.lgb"
PARAM_PATH = "best_params.json"
SEED = 42

def load_cached_data():
    """キャッシュ済み全銘柄を統合"""
    dfs = []
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(CACHE_DIR, f))
            dfs.append(df)
    if not dfs:
        raise RuntimeError("No cached data found in price_cache/")
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df = all_df.sort_values(["date","ticker"]).reset_index(drop=True)
    return all_df

def make_features(df):
    """特徴量生成"""
    df = df.copy()
    df["ret1"] = df.groupby("ticker")["close"].pct_change()
    df["vol_chg"] = df.groupby("ticker")["volume"].pct_change()
    df["ma5"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(5).mean())
    df["ma20"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20).mean())
    df["mom5"] = df.groupby("ticker")["close"].transform(lambda x: x.pct_change(5))
    df["label"] = (df.groupby("ticker")["close"].shift(-5) / df["close"] - 1).clip(-0.2, 0.2)
    df = df.dropna().reset_index(drop=True)
    return df

def train_final_model(df, params):
    """LightGBM モデル学習 (ウォークフォワード交差)"""
    feats = ["ret1","vol_chg","ma5","ma20","mom5"]
    X, y = df[feats], (df["label"] > 0).astype(int)

    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []
    models = []
    for i, (tr, te) in enumerate(tscv.split(X)):
        dtrain = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
        dtest = lgb.Dataset(X.iloc[te], label=y.iloc[te])
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtest],
            callbacks=[lgb.log_evaluation(period=0)]  # 👈 verbose_eval=False の代わり
        )
        preds = model.predict(X.iloc[te])
        auc = roc_auc_score(y.iloc[te], preds)
        aucs.append(auc)
        models.append(model)
        print(f"  Fold{i+1}: AUC={auc:.4f}")
    print(f"[done] Mean AUC={np.mean(aucs):.4f}")
    return models[-1]


if __name__ == "__main__":
    if not os.path.exists(PARAM_PATH):
        raise FileNotFoundError(f"{PARAM_PATH} not found. Run parameta.py first.")

    print("[load] cached data ...")
    df = load_cached_data()

    print("[load] params ...")
    params = json.load(open(PARAM_PATH))
    params.update({
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_threads": 4,
        "seed": SEED
    })

    print("[train] training model ...")
    feats_df = make_features(df)
    model = train_final_model(feats_df, params)

    print(f"[save] model -> {MODEL_PATH}")
    model.save_model(MODEL_PATH)
    print("✅ model_best.lgb saved successfully!")
