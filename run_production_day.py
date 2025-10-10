# -*- coding: utf-8 -*-
"""
AI株式シミュレーター（翌寄付き約定版・日次CSVログ付き・売却フォールバック）
- 新規買い / 売却 は原則「翌取引日の始値（open）」で約定
- SLIPPAGE を導入（買いは 1+SLIPPAGE、売りは 1-SLIPPAGE）
- 売却時に当日/過去の価格が無ければフォールバックで利用
- equity_state.json / equity_log.csv を更新
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

# SLIPPAGE: 約定での価格悪化（例 0.0002 = 0.02%）
SLIPPAGE = 0.0002

# CSV ログ設定
ENABLE_CSV_LOG = True
LOG_FILE = Path("equity_log.csv")

# Discord
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
STATE_FILE = Path("equity_state.json")

ALL_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AMD","NFLFX","ADBE",
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
# ユーティリティ関数（特徴量等）
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
            if isinstance(data.columns, pd.MultiIndex):
                if t not in data.columns.levels[0]:
                    print(f"⚠ データ無し: {t}")
                    continue
                df = data[t].copy().reset_index()
            else:
                df = data.copy().reset_index()
            df.columns = [c.lower() for c in df.columns]
            need = {"date","open","high","low","close","volume"}
            if not need.issubset(set(df.columns)):
                print(f"⚠ 欠損列: {t}")
                continue
            feat = make_features(df, t)
            out.append(feat)
        except Exception as e:
            print(f"⚠ {t}: {e}")
            continue
    if not out:
        raise RuntimeError("No data fetched.")
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
        d = df.dropna(subset=[f"label{k}"])
        if d.empty:
            models[k] = None
            continue
        X, y = d[FEATS], d[f"label{k}"]
        models[k] = lgb.train(params, lgb.Dataset(X, label=y))
    return models

def predict_models(models, df):
    X = df[FEATS]
    out = {}
    for k in ["1","3","5"]:
        if models.get(k) is None:
            out[f"p{k}"] = np.zeros(len(df))
        else:
            out[f"p{k}"] = models[k].predict(X)
    return out

def estimate_mu(df):
    mu = {}
    for k in ["1","3","5"]:
        pos = df[f"ret_next{k}"].loc[df[f"label{k}"]==1]
        mu[k] = pos.mean() if len(pos)>0 else 0.0
    return mu

# =========================
# 状態管理 / Discord / CSV logging
# =========================
def load_state():
    if not STATE_FILE.exists():
        return {"capital": INIT_CAPITAL, "positions": []}
    return json.load(open(STATE_FILE, "r", encoding="utf-8"))

def save_state(state):
    json.dump(state, open(STATE_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

def notify_discord(msg):
    if not DISCORD_WEBHOOK_URL:
        print(msg); return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
    except Exception as e:
        print(f"Discord通知失敗: {e}")

def append_daily_log(run_date_jst, market_date, realized, unrealized, equity, capital, positions, buys, sells):
    if not ENABLE_CSV_LOG:
        return
    pos_summary = []
    for p in positions:
        pos_summary.append(f"{p['ticker']}:{p['hold_days']}/{p.get('held_days',0)}:{int(p['amount'])}")
    pos_summary_str = ";".join(pos_summary)
    buys_str = ";".join(buys) if buys else ""
    sells_str = ";".join(sells) if sells else ""

    prev_equity = None
    if LOG_FILE.exists():
        try:
            prev = pd.read_csv(LOG_FILE)
            if len(prev)>0:
                prev_equity = float(prev.iloc[-1]["equity"])
        except Exception:
            prev_equity = None

    if prev_equity is None or prev_equity == 0:
        daily_ret = 0.0
    else:
        daily_ret = (equity / prev_equity - 1.0) * 100.0

    row = {
        "run_date_jst": run_date_jst.isoformat(),
        "market_date": pd.to_datetime(market_date).date().isoformat(),
        "realized_pnl": float(realized),
        "unrealized_pnl": float(unrealized),
        "equity": float(equity),
        "capital": float(capital),
        "num_positions": len(positions),
        "positions": pos_summary_str,
        "buys": buys_str,
        "sells": sells_str,
        "daily_return_pct": float(daily_ret)
    }

    dfrow = pd.DataFrame([row])
    header = not LOG_FILE.exists()
    dfrow.to_csv(LOG_FILE, mode="a", index=False, header=header)

# =========================
# 売却フォールバック用ヘルパー
# =========================
def get_last_available_price(all_df_local, ticker, date):
    sub = all_df_local[(all_df_local["ticker"]==ticker) & (all_df_local["date"]<=date)].sort_values("date", ascending=False)
    if not sub.empty:
        return float(sub.iloc[0]["close"])
    return None

# =========================
# シミュレーション（翌寄付き約定ロジック）
# =========================
def simulate_continuous(all_df):
    df = all_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    unique_dates = sorted(df["date"].unique())
    market_date = df["date"].max()           # シグナル算出に使う最新市場データ日
    # next trading date (for next-open execution), if exists
    try:
        idx_today = unique_dates.index(pd.Timestamp(market_date))
        next_date = unique_dates[idx_today + 1] if idx_today + 1 < len(unique_dates) else None
    except ValueError:
        next_date = None

    tr = df[df["date"] >= market_date - pd.DateOffset(months=6)]
    te = df[df["date"] == market_date]
    df_next = df[df["date"] == next_date] if next_date is not None else pd.DataFrame(columns=df.columns)

    if tr.empty or te.empty:
        print("⚠ データ不足")
        return

    # マップ化（高速アクセス用）
    te_close_map = {r.ticker: float(r.close) for r in te.itertuples()}
    te_open_map = {r.ticker: float(r.open) for r in te.itertuples()}
    next_open_map = {r.ticker: float(r.open) for r in df_next.itertuples()} if not df_next.empty else {}
    next_close_map = {r.ticker: float(r.close) for r in df_next.itertuples()} if not df_next.empty else {}

    state = load_state()
    capital = state.get("capital", INIT_CAPITAL)
    old_positions = state.get("positions", [])
    realized_pnl = 0.0
    sold_lines = []
    sells_for_log = []
    sells_skipped_for_log = []
    remaining = []

    # ---------- 売却処理（保有満了→原則 next_open で約定、なければフォールバック） ----------
    for pos in old_positions:
        pos["held_days"] = pos.get("held_days", 0) + 1
        if pos["held_days"] >= pos["hold_days"]:
            sell_price = None
            sell_source = None
            # 優先：翌営業日の始値（next_open）
            if next_date is not None and pos["ticker"] in next_open_map:
                sell_price = next_open_map[pos["ticker"]] * (1.0 - SLIPPAGE)
                sell_source = f"next_open({next_date.date()})"
            # 次の優先：当日の終値（te_close）
            elif pos["ticker"] in te_close_map:
                sell_price = te_close_map[pos["ticker"]] * (1.0 - SLIPPAGE)
                sell_source = f"te_close({market_date.date()})"
            else:
                # フォールバック：全データから直近利用可能な close を使う
                last_price = get_last_available_price(df, pos["ticker"], market_date)
                if last_price is not None:
                    sell_price = last_price * (1.0 - SLIPPAGE)
                    sell_source = "last_available_before_or_on_market_date"
                else:
                    sell_price = None
                    sell_source = "no_price_found"

            if sell_price is None:
                # 価格が無くて売却できない → 保有継続
                remaining.append(pos)
                sells_skipped_for_log.append(f"{pos['ticker']}:skip_no_price")
                continue

            # 利益計算（手数料を考慮）
            shares = pos["amount"] / pos["buy_price"]
            proceeds = shares * sell_price
            buy_comm = COMMISSION * pos["amount"]
            sell_comm = COMMISSION * proceeds
            profit = proceeds - pos["amount"] - buy_comm - sell_comm

            realized_pnl += profit
            sold_lines.append(f"{pos['ticker']} ({pos['hold_days']}日): {((sell_price/pos['buy_price']-1)*100):+.2f}% ({profit:+,.0f}円) [{sell_source}]")
            sells_for_log.append(f"{pos['ticker']}:profit={profit:+.0f}:src={sell_source}")
        else:
            remaining.append(pos)

    # スキップログを統合
    if sells_skipped_for_log:
        sells_for_log.extend(sells_skipped_for_log)

    capital += realized_pnl

    # ---------- 新規購入（シグナルに対し next_open を優先して約定価格を設定） ----------
    add_future_labels_inplace(tr)
    add_future_labels_inplace(te)
    models = train_models(tr)
    mu = estimate_mu(tr)
    preds = predict_models(models, te)

    scores = np.vstack([preds["p1"]*mu["1"], preds["p3"]*mu["3"], preds["p5"]*mu["5"]]).T
    best_score = scores.max(axis=1)
    te["score"] = best_score
    thr = np.quantile(best_score, FIXED_SCORE_QUANTILE)
    buys_df = te[te["score"] >= thr].sort_values("score", ascending=False).head(TOP_K)

    new_positions = []
    buys_for_log = []
    if not buys_df.empty:
        per_trade = capital * PER_POS_FRAC / len(buys_df)
        for i, row in enumerate(buys_df.itertuples()):
            p1, p3, p5 = preds["p1"][i], preds["p3"][i], preds["p5"][i]
            idx = np.argmax([p1*mu["1"], p3*mu["3"], p5*mu["5"]])
            hold_days = [1,3,5][idx]

            # 約定価格の決定（優先: next_open -> 当日終値 -> 直近 available）
            buy_price = None
            buy_source = None
            if next_date is not None and row.ticker in next_open_map:
                buy_price = next_open_map[row.ticker] * (1.0 + SLIPPAGE)
                buy_source = f"next_open({next_date.date()})"
            elif row.ticker in te_close_map:
                buy_price = te_close_map[row.ticker] * (1.0 + SLIPPAGE)
                buy_source = f"te_close({market_date.date()})"
            else:
                last_price = get_last_available_price(df, row.ticker, market_date)
                if last_price is not None:
                    buy_price = last_price * (1.0 + SLIPPAGE)
                    buy_source = "last_available_before_or_on_market_date"
                else:
                    # 価格が取得できない銘柄はスキップ（安全）
                    continue

            new_positions.append({
                "ticker": row.ticker,
                "buy_price": float(buy_price),
                "amount": float(per_trade),
                "hold_days": int(hold_days),
                "held_days": 0
            })
            buys_for_log.append(f"{row.ticker}:hold={hold_days}:price={buy_price:.2f}:src={buy_source}")

    all_positions = remaining + new_positions

    # 含み評価（評価には market_date の終値を用いる。必要なら next_close_map に変更可能）
    unrealized = 0.0
    for pos in all_positions:
        cur_price = te_close_map.get(pos["ticker"], pos["buy_price"])
        unrealized += (cur_price / pos["buy_price"] - 1.0) * pos["amount"]

    current_equity = capital + unrealized

    # 状態保存（capital は確定残高）
    state = {"capital": capital, "positions": all_positions}
    save_state(state)

    # 通知作成（JSTベースの実行日）
    run_date_jst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).date()
    sold_str = "\n".join(sold_lines) if sold_lines else "（売却なし）"
    buy_str = "\n".join([f"{p['ticker']} ({p['hold_days']}日保有) @ {p['buy_price']:.2f}" for p in new_positions]) if new_positions else "（新規購入なし）"

    msg = (
        f"📅 **{run_date_jst} トレード結果**\n"
        f"**市場データ日:** {market_date.date()}\n"
        f"**約定想定日（next trading date）:** {next_date.date() if next_date is not None else 'N/A'}\n"
        f"**売却:**\n{sold_str}\n"
        f"**新規購入:**\n{buy_str}\n"
        f"**実現損益:** {realized_pnl:+,.0f}円\n"
        f"**含み損益:** {unrealized:+,.0f}円\n"
        f"**評価残高（含み込み）:** {current_equity:,.0f}円\n"
        f"**確定残高（現金）:** {capital:,.0f}円"
    )
    print(msg)
    notify_discord(msg)

    # CSV ログ書き込み
    append_daily_log(run_date_jst=run_date_jst,
                     market_date=market_date,
                     realized=realized_pnl,
                     unrealized=unrealized,
                     equity=current_equity,
                     capital=capital,
                     positions=all_positions,
                     buys=buys_for_log,
                     sells=sells_for_log)

# =========================
# メイン実行（土日スキップ）
# =========================
def main():
    # 土日スキップ（JSTベース）
    weekday = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).weekday()  # 月=0, 日=6
    if weekday >= 5:
        msg = f"🛌 {datetime.date.today()} は休場日のためスキップ"
        print(msg)
        notify_discord(msg)
        return

    all_df = load_all_data_fast(ALL_TICKERS)
    simulate_continuous(all_df)

if __name__ == "__main__":
    main()














