# ============================================================
#  generate_model_best.py
#  年率32%時のLightGBMパラメータで固定モデルを学習
#  出力: model_best.lgb
# ============================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import yfinance as yf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from feature_engineer import make_features

# ============================================================
# 設定
# ============================================================
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "JPM", "XOM", "PG"]
START = "2018-01-01"
END   = None  # 現在まで
LABEL_HORIZON = 5  # 5日後の上昇確率を学習

# 年率32%達成時のパラメータ固定
BEST_PARAMS = dict(
    objective="binary",
    boosting_type="gbdt",
    learning_rate=0.07,
    num_leaves=42,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    lambda_l2=0.3,
    min_child_samples=25,
    n_estimators=400,
    verbosity=-1
)

# ============================================================
# データ取得と特徴量生成
# ============================================================
def load_all_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, group_by='ticker', threads=True, progress=False)
    all_df = []
    for t in tickers:
        if t not in data.columns.levels[0]:
            print(f"[warn] missing: {t}")
            continue
        df = data[t].copy().reset_index()
        df = make_features(df)
        df["ticker"] = t
        all_df.append(df)
    all_df = pd.concat(all_df, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["Date"])
    all_df = all_df.drop(columns=["Date"], errors="ignore")
    return all_df

def add_future_labels(df, horizon=5):
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)
    df["future_close"] = df.groupby("ticker")["Close"].shift(-horizon)
    df["label1"] = (df["future_close"] > df["Close"]).astype(int)
    df = df.dropna().reset_index(drop=True)
    return df

# ============================================================
# 学習データ準備
# ============================================================
print("[load] downloading data ...")
df = load_all_data(TICKERS, START, END)
df = add_future_labels(df, LABEL_HORIZON)

FEATS = [c for c in df.columns if c not in ["date","ticker","future_close","label1"]]
X, y = df[FEATS], df["label1"]

# ============================================================
# 学習（時系列分割で安定化）
# ============================================================
print("[train] training model ...")
tscv = TimeSeriesSplit(n_splits=5)
models, aucs = [], []
for i, (tr, te) in enumerate(tscv.split(X)):
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[te]
    model = lgb.train(BEST_PARAMS, lgb.Dataset(Xtr, label=ytr))
    pred = model.predict(Xte)
    auc = roc_auc_score(yte, pred)
    aucs.append(auc)
    models.append(model)
    print(f"  Fold{i+1}: AUC={auc:.4f}")

print(f"[done] Mean AUC={np.mean(aucs):.4f}")

# ============================================================
# モデル保存（最終fold）
# ============================================================
models[-1].save_model("model_best.lgb")
print("✅ model_best.lgb saved successfully!")
