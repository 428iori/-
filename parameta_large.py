# parameta_large.py
# 大規模学習・最適化フルコード（CSVキャッシュ＆ウォークフォワードCV＆Optuna＆LightGBM）
# ------------------------------------------------------------
# 特徴:
#  - yfinanceで取得→ price_cache/{TICKER}.csv にキャッシュ（以後はファイル読込）
#  - 特徴量20+（RSI/MACD/Bollinger/momentum/volなど、pandasのみで計算）
#  - ラベル: 1,3,5日先の上昇(>0)をバイナリ化。学習は label1 を使用（高速・汎用）
#  - 評価: 36ヶ月学習→6ヶ月評価のローリングで AUC の平均を最大化
#  - OptunaでLGBMハイパラ探索 → best_params.json / model_best.lgb を出力
#  - 依存を最低限に（parquet系は使わない）。429等は軽いリトライ実装。
# ------------------------------------------------------------

import os, sys, time, json, math, argparse, warnings, random
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

CACHE_DIR = Path("price_cache")
RESULT_DIR = Path("results")
CACHE_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# =========================
# デフォルトのユニバース（必要ならtickers_fileで差し替え）
# =========================
DEFAULT_TICKERS = [
    # mega tech & semis
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AMD","INTC","AVGO",
    # software / cloud / cyber
    "ADBE","CRM","NOW","PANW","DDOG","SNOW","CRWD","ZS","NET","MDB",
    # consumer / retail
    "COST","WMT","HD","LOW","NKE","MCD","SBUX","KO","PEP","PG",
    # travel / leisure / EV
    "BKNG","ABNB","UBER","LYFT","RBLX","NFLX","DIS","F","GM","CAT",
    # energy / materials / industrials
    "XOM","CVX","PSX","MPC","OXY","SLB","EOG","FCX","NEM","HON",
    # defense
    "LMT","RTX","NOC","GD",
    # banks / fin
    "JPM","BAC","C","MS","GS","BLK","SCHW","SPGI","V","MA","AXP",
    # healthcare
    "JNJ","PFE","MRK","BMY","LLY","UNH","CI","HUM","CVS","DHR","TMO","MDT",
    # misc quality
    "ORCL","QCOM","TXN","CSCO","SHOP","PYPL","SQ","IBM","GE","DE"
]

# -------------------------
# 便利関数: RSI/MACD/BBANDS など（pandasのみで実装）
# -------------------------
def rsi(series: pd.Series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_dn = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_dn + 1e-12)
    return 100 - 100 / (1 + rs)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_z(close: pd.Series, n=20, k=2):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper = ma + k*sd
    lower = ma - k*sd
    z_up = (close - upper) / (sd*2 + 1e-12)
    z_low = (close - lower) / (sd*2 + 1e-12)
    return z_up, z_low

# -------------------------
# yfinanceダウンロード→CSVキャッシュ
# -------------------------
def yf_download_single(ticker, start, end, max_retries=3, pause=1.0):
    for i in range(1, max_retries+1):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                             progress=False, group_by='ticker', threads=False)
            if df is None or len(df) == 0:
                raise RuntimeError("empty df")
            # MultiIndex対応 or 単層対応
            if isinstance(df.columns, pd.MultiIndex):
                # 1ティッカー時は ('Close', 'TICKER') などになることがある
                if 'Close' in df.columns.get_level_values(0):
                    close = df['Close']
                    # 1カラム or 複数? 単ティッカーのはずなので取り出す
                    if isinstance(close, pd.DataFrame) and close.shape[1] == 1:
                        df = pd.DataFrame({
                            "Date": df.index,
                            "Open": df[('Open', ticker)] if ('Open', ticker) in df.columns else df['Open'].iloc[:,0],
                            "High": df[('High', ticker)] if ('High', ticker) in df.columns else df['High'].iloc[:,0],
                            "Low":  df[('Low',  ticker)] if ('Low',  ticker) in df.columns else df['Low'].iloc[:,0],
                            "Close":df[('Close',ticker)] if ('Close',ticker) in df.columns else df['Close'].iloc[:,0],
                            "Volume":df[('Volume',ticker)] if ('Volume',ticker) in df.columns else df['Volume'].iloc[:,0],
                        })
                    else:
                        # 念のためのフォールバック
                        df = df.droplevel(1, axis=1).reset_index()
                else:
                    df = df.reset_index()
            else:
                df = df.reset_index()
            # 必須列チェック
            need = {"Date","Open","High","Low","Close","Volume"}
            if not need.issubset(set(df.columns)):
                raise RuntimeError("missing columns")
            df = df.rename(columns=str.lower)
            df["ticker"] = ticker
            return df
        except Exception:
            if i == max_retries:
                return None
            time.sleep(pause * i)
    return None

def load_or_fetch(ticker, start, end, verbose=False):
    fp = CACHE_DIR / f"{ticker}.csv"
    if fp.exists():
        try:
            df = pd.read_csv(fp)
            # 最低限のチェック
            if {"date","open","high","low","close","volume","ticker"}.issubset(df.columns):
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                return df.dropna(subset=["date"]).reset_index(drop=True)
        except Exception:
            pass  # 壊れている場合は再DL
    if verbose:
        print(f"[dl] {ticker} ... ", end="", flush=True)
    df = yf_download_single(ticker, start, end)
    if df is None or len(df)==0:
        if verbose: print("fail")
        return None
    df.to_csv(fp, index=False)
    if verbose: print(f"ok rows={len(df)}")
    return df

def load_universe(tickers, start, end, verbose=True):
    all_dfs = []
    for t in tickers:
        df = load_or_fetch(t, start, end, verbose=verbose)
        if df is not None and len(df)>0:
            all_dfs.append(df)
    if not all_dfs:
        raise RuntimeError("No price data fetched. Check tickers/start/end/network.")
    all_df = pd.concat(all_dfs, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"], errors="coerce")
    all_df = all_df.dropna(subset=["date"])
    all_df = all_df.sort_values(["date","ticker"]).reset_index(drop=True)
    return all_df

# -------------------------
# マーケット指標（SPY/QQQ/^VIX）→単純なリターン特徴へ
# -------------------------
def load_market_index(start, end, verbose=True):
    idx_list = ["SPY","QQQ","^VIX"]
    try:
        data = yf.download(idx_list, start=start, end=end, auto_adjust=True,
                           group_by='ticker', threads=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            close = data.xs("Close", axis=1, level=1) if "Close" in data.columns.get_level_values(1) else data.xs("Close", axis=1, level=0)
        else:
            # まれに単層で来るケース
            close = data["Close"].to_frame()
        close = close.rename(columns={"SPY":"spy","QQQ":"qqq","^VIX":"vix"})
        rets = close.pct_change().rename(columns={"spy":"spy_ret","qqq":"qqq_ret","vix":"vix_ret"}).reset_index()
        # 'Date' or 'index' のどちらか
        if "Date" in rets.columns:
            rets = rets.rename(columns={"Date":"date"})
        else:
            rets = rets.rename(columns={"index":"date"})
        rets["date"] = pd.to_datetime(rets["date"], errors="coerce")
        return rets.dropna(subset=["date"])
    except Exception:
        if verbose:
            print("[warn] market index download failed; continuing without it")
        return pd.DataFrame(columns=["date","spy_ret","qqq_ret","vix_ret"])

# -------------------------
# 特徴量生成
# -------------------------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: [date, open, high, low, close, volume, ticker]"""
    df = df.copy().sort_values(["ticker","date"])
    g = df.groupby("ticker", group_keys=False)

    # basic returns / momentum
    df["ret1"]  = g["close"].pct_change()
    df["ret3"]  = g["close"].pct_change(3)
    df["ret5"]  = g["close"].pct_change(5)
    df["ret10"] = g["close"].pct_change(10)
    df["mom20"] = g["close"].apply(lambda x: x / x.shift(20) - 1)

    # moving averages
    df["sma10"] = g["close"].transform(lambda x: x.rolling(10).mean())
    df["sma30"] = g["close"].transform(lambda x: x.rolling(30).mean())
    df["sma60"] = g["close"].transform(lambda x: x.rolling(60).mean())
    df["ma_gap"] = (df["close"] - df["sma30"]) / (df["sma30"] + 1e-12)

    # volatility
    df["std20"] = g["close"].transform(lambda x: x.pct_change().rolling(20).std())
    df["atr14"] = g[["high","low","close"]].apply(
        lambda x: (x["high"]-x["low"]).rolling(14).mean(), raw=False
    ).reset_index(level=0, drop=True)

    # RSI / MACD / Bollinger
    df["rsi14"] = g["close"].transform(lambda x: rsi(x, 14))
    macd_line, signal_line, hist = macd(g["close"].apply(lambda s: s).reset_index(level=0, drop=True))
    df["macd"] = macd_line.values
    df["macd_sig"] = signal_line.values
    df["bb_up"], df["bb_low"] = bollinger_z(df["close"], 20, 2)

    # volume
    df["vol_chg"] = g["volume"].pct_change()
    df["vol_ratio"] = g["volume"].transform(lambda x: x / (x.rolling(20).mean() + 1e-12))

    # drawdown (20日高値からの下落率)
    df["dd20"] = g["close"].transform(lambda x: x / x.rolling(20).max() - 1)

    return df

FEATS = [
    "ret1","ret3","ret5","ret10","mom20",
    "sma10","sma30","sma60","ma_gap",
    "std20","atr14","rsi14","macd","macd_sig","bb_up","bb_low",
    "vol_chg","vol_ratio","dd20",
    # マーケット指標があればこれも
    "spy_ret","qqq_ret","vix_ret"
]

# -------------------------
# ラベル作成（未来情報はWFのtrain内でのみ）
# -------------------------
def add_future_labels_inplace(df: pd.DataFrame):
    g = df.groupby("ticker")
    for n in [1,3,5]:
        df[f"ret_next{n}"] = g["close"].shift(-n)/df["close"] - 1
        df[f"label{n}"] = (df[f"ret_next{n}"] > 0).astype(int)
    return df

# -------------------------
# LightGBMの学習 & WF評価
# -------------------------
def train_lgbm(X, y, params):
    dtrain = lgb.Dataset(X, label=y)
    # シンプルに学習（verbose_evalはv4で廃止）
    model = lgb.train(params, dtrain)
    return model

def walk_forward_auc(all_df: pd.DataFrame, params, train_months=36, test_months=6, label_col="label1"):
    df = all_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    months = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")[train_months:]
    if len(months)==0:
        raise RuntimeError("not enough months for WF")

    aucs = []
    for ms in months:
        train_start = ms - pd.DateOffset(months=train_months)
        train_end   = ms - pd.DateOffset(days=1)
        test_start  = ms
        test_end    = ms + pd.DateOffset(months=test_months) - pd.DateOffset(days=1)

        tr = df[(df["date"]>=train_start)&(df["date"]<=train_end)].copy()
        te = df[(df["date"]>=test_start)&(df["date"]<=test_end)].copy()
        if len(tr)<500 or len(te)<100:
            continue

        # ラベルはtrain/testで個別に作る（リーク防止のため、本来はtrainだけでfitでもOKだが簡易化）
        add_future_labels_inplace(tr)
        add_future_labels_inplace(te)

        # 特徴に欠損がある行を除外
        tr = tr.dropna(subset=FEATS+[label_col])
        te = te.dropna(subset=FEATS+[label_col])
        if len(tr)==0 or len(te)==0:
            continue

        Xtr, ytr = tr[FEATS], tr[label_col].astype(int)
        Xte, yte = te[FEATS], te[label_col].astype(int)

        model = train_lgbm(Xtr, ytr, params)
        p = model.predict(Xte)
        # AUC
        try:
            auc = roc_auc_score(yte, p)
            aucs.append(auc)
        except Exception:
            continue

    if not aucs:
        return -9999.0, []  # 失敗扱い
    return float(np.mean(aucs)), aucs

# -------------------------
# Optuna 目的関数
# -------------------------
def suggest_params(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": -1,
        "random_state": 42,
    }
    return params

# -------------------------
# メイン: データ→特徴量→WF→Optuna→保存
# -------------------------
def run_optuna_search(tickers, start, end, n_trials_local=50, seed=42, jobs=1, verbose=True):
    if start is None or end is None:
        raise RuntimeError("start/end is required (YYYY-MM-DD).")

    # 1) データ収集（キャッシュ）
    if verbose:
        print(f"[load] universe ... ({len(tickers)} tickers)")
    px = load_universe(tickers, start, end, verbose=verbose)

    # 2) マーケット指標
    mkt = load_market_index(start, end, verbose=verbose)
    if not mkt.empty:
        px = px.merge(mkt, on="date", how="left")
    else:
        for c in ["spy_ret","qqq_ret","vix_ret"]:
            if c not in px.columns:
                px[c] = 0.0

    # 3) 特徴量
    fx = make_features(px).dropna(subset=["close"])
    fx = fx.dropna(subset=FEATS).reset_index(drop=True)

    # 4) Optuna
    def objective(trial):
        params = suggest_params(trial)
        score, aucs = walk_forward_auc(fx, params,
                                       train_months=36, test_months=6, label_col="label1")
        # 最大化したいので score をそのまま返す（失敗時は -9999 が返る）
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    if verbose:
        print(f"[optuna] optimizing ... trials: {n_trials_local}")
    study.optimize(objective, n_trials=n_trials_local, n_jobs=jobs, show_progress_bar=True)

    # 5) 結果保存
    trials_df = study.trials_dataframe(attrs=("number","value","params","state","datetime_start","datetime_complete"))
    trials_csv = RESULT_DIR / "optuna_trials_summary.csv"
    trials_df.to_csv(trials_csv, index=False)

    best_params = study.best_trial.params
    best_params_file = RESULT_DIR / "best_params.json"
    with open(best_params_file, "w") as f:
        json.dump(best_params, f, indent=2)

    # 6) ベストパラメータで最終モデル学習（全期間・全銘柄）
    if verbose:
        print("[final] training final model with best params ...")
    # ラベル付与→欠損除去
    fx2 = fx.copy()
    add_future_labels_inplace(fx2)
    fx2 = fx2.dropna(subset=FEATS+["label1"]).reset_index(drop=True)
    Xall = fx2[FEATS]
    yall = fx2["label1"].astype(int)
    final_params = {
        "objective": "binary", "metric": "auc",
        "boosting_type": "gbdt", "verbosity": -1, "random_state": 42,
        **best_params
    }
    model = train_lgbm(Xall, yall, final_params)

    model_file = RESULT_DIR / "model_best.lgb"
    model.save_model(str(model_file))

    # 7) 追加メトリクス
    best_value = float(study.best_value) if study.best_value is not None else float("nan")
    metrics = {
        "best_auc_mean": best_value,
        "n_trials": n_trials_local,
        "start": start, "end": end,
        "n_rows": int(len(fx2)),
        "n_tickers": int(fx2["ticker"].nunique())
    }
    with open(RESULT_DIR / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"[done] AUC(best)={best_value:.4f}")
        print(f"[save] {trials_csv}")
        print(f"[save] {best_params_file}")
        print(f"[save] {model_file}")
        print(f"[save] results/metrics_summary.json")

    return study

# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Large-scale parameter search with WF-CV (LightGBM + Optuna)")
    ap.add_argument("--start", type=str, default="2010-01-01")
    ap.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (None=today)")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=1, help="Optuna parallel jobs")
    ap.add_argument("--tickers_file", type=str, default=None, help="path to file with 1 ticker per line")
    return ap.parse_args()

def resolve_end(e):
    if e is None or str(e).lower()=="none":
        # 直近日付（UTC基準の当日でOK）
        return datetime.utcnow().strftime("%Y-%m-%d")
    return e

if __name__ == "__main__":
    args = parse_args()
    START = args.start
    END = resolve_end(args.end)

    # ユニバース決定
    if args.tickers_file and os.path.exists(args.tickers_file):
        with open(args.tickers_file) as f:
            tickers = [ln.strip() for ln in f if ln.strip()]
    else:
        tickers = DEFAULT_TICKERS

    print(f"[run] tickers={len(tickers)}, START={START}, END={END}, trials={args.trials}, jobs={args.jobs}")
    study = run_optuna_search(
        tickers=tickers,
        start=START,
        end=END,
        n_trials_local=args.trials,
        seed=42,
        jobs=args.jobs,
        verbose=True
    )
