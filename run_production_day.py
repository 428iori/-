# -*- coding: utf-8 -*-
"""
AI株式シミュレーター（翌寄付き約定 + 売却ログ強化 + 日次CSV）
- 翌寄付き(next-open)を優先して約定。なければ当日終値、さらに過去価格をフォールバック。
- 売却が実行された or スキップされた理由を必ず通知/CSVに残すパッチ適用済み
- 土日スキップ（JST）
"""
import os, json, datetime, warnings, requests
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# =========================
# 設定（調整可）
# =========================
START = "2015-01-01"
END = None
RANDOM_SEED = 42
TRAIN_MONTHS = 6
TOP_K = 3
INIT_CAPITAL = 1_000_000
PER_POS_FRAC = 0.20
COMMISSION = 0.0005
FIXED_SCORE_QUANTILE = 0.50
SOFTMAX_TEMP = 0.15
RET_CLIP_LOW, RET_CLIP_HIGH = -0.08, 0.30

# SLIPPAGE: 約定での価格悪化（例 0.0002 = 0.02%）
SLIPPAGE = 0.0002

# ログ/ファイル
ENABLE_CSV_LOG = True
LOG_FILE = Path("equity_log.csv")
STATE_FILE = Path("equity_state.json")

# Discord
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# デバッグ（本番は False）
DEBUG = False

# ユニバース（必要に応じて編集）
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
# ヘルパー（特徴量等）
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
    c = df["close"].astype(float)
    v = df["volume"].astype(float) if "volume" in df.columns else pd.Series(np.nan, index=df.index)
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

# =========================
# データ取得
# =========================
def load_all_data_fast(tickers):
    print(f"Downloading {len(tickers)} tickers...")
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, group_by='ticker', threads=True, progress=False)
    out = []
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t not in raw.columns.levels[0]:
                    if DEBUG: print(f"[WARN] no data for {t}")
                    continue
                df = raw[t].copy().reset_index()
            else:
                df = raw.copy().reset_index()
            df.columns = [c.lower() for c in df.columns]
            need = {"date","open","high","low","close","volume"}
            if not need.issubset(set(df.columns)):
                if DEBUG: print(f"[WARN] missing cols for {t}")
                continue
            feat = make_features(df, t)
            out.append(feat)
        except Exception as e:
            if DEBUG: print(f"[ERR] {t}: {e}")
            continue
    if not out:
        raise RuntimeError("No data fetched.")
    all_df = pd.concat(out, ignore_index=True).sort_values(["date","ticker"]).reset_index(drop=True)
    print(f"Data rows: {len(all_df)}  period: {all_df['date'].min().date()} ~ {all_df['date'].max().date()}")
    return all_df

# =========================
# ラベル / モデル
# =========================
def add_future_labels_inplace(df):
    for n in [1,3,5]:
        df[f"ret_next{n}"] = df.groupby("ticker")["close"].shift(-n) / df["close"] - 1
        df[f"label{n}"] = (df[f"ret_next{n}"] > 0).astype(int)
    return df

def train_models(df):
    params = {"objective":"binary","metric":"auc","random_state":RANDOM_SEED,
              "learning_rate":0.05,"num_leaves":31,"n_estimators":400,"verbosity":-1}
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
# 状態管理 / 通知 / CSV
# =========================
def load_state():
    if not STATE_FILE.exists():
        return {"capital": INIT_CAPITAL, "positions": []}
    try:
        return json.load(open(STATE_FILE, "r", encoding="utf-8"))
    except Exception:
        return {"capital": INIT_CAPITAL, "positions": []}

def save_state(state):
    json.dump(state, open(STATE_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

def notify_discord(msg):
    if not DISCORD_WEBHOOK_URL:
        print(msg)
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg}, timeout=5)
    except Exception as e:
        print(f"[WARN] Discord notify failed: {e}")
        print(msg)

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

    daily_ret = 0.0
    if prev_equity is not None and prev_equity != 0:
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
# 売却フォールバックヘルパー
# =========================
def get_last_available_price(all_df_local, ticker, date):
    sub = all_df_local[(all_df_local["ticker"]==ticker) & (all_df_local["date"]<=date)].sort_values("date", ascending=False)
    if not sub.empty:
        return float(sub.iloc[0]["close"])
    return None

# =========================
# メインロジック（翌寄付き約定・売却ログ強化）
# =========================
def simulate_continuous(all_df):
    df = all_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    unique_dates = sorted(df["date"].unique())
    market_date = df["date"].max()

    # next trading date detection
    try:
        idx_today = unique_dates.index(pd.Timestamp(market_date))
        next_date = unique_dates[idx_today + 1] if idx_today + 1 < len(unique_dates) else None
    except ValueError:
        next_date = None

    # If next_date missing, attempt to fetch a couple extra calendar days via yfinance
    df_next = pd.DataFrame(columns=df.columns)
    if next_date is None:
        try:
            extra_start = (pd.to_datetime(market_date) + pd.Timedelta(days=1)).date().isoformat()
            extra_end   = (pd.to_datetime(market_date) + pd.Timedelta(days=4)).date().isoformat()
            extra_raw = yf.download(ALL_TICKERS, start=extra_start, end=extra_end,
                                    auto_adjust=True, group_by='ticker', threads=True, progress=False)
            extra_list = []
            for t in ALL_TICKERS:
                try:
                    if isinstance(extra_raw.columns, pd.MultiIndex):
                        if t not in extra_raw.columns.levels[0]:
                            continue
                        tmp = extra_raw[t].reset_index()
                    else:
                        tmp = extra_raw.reset_index()
                    tmp.columns = [c.lower() for c in tmp.columns]
                    if not {"date","open","close"}.issubset(set(tmp.columns)):
                        continue
                    tmp = tmp[["date","open","high","low","close","volume"]].copy()
                    tmp["ticker"] = t
                    extra_list.append(tmp)
                except Exception:
                    continue
            if extra_list:
                extra_df = pd.concat(extra_list, ignore_index=True)
                extra_df["date"] = pd.to_datetime(extra_df["date"])
                cand_dates = sorted(extra_df[extra_df["date"]>pd.to_datetime(market_date)]["date"].unique())
                if len(cand_dates)>0:
                    next_date = cand_dates[0]
                    df_next = extra_df[extra_df["date"]==next_date].copy()
        except Exception as e:
            if DEBUG: print(f"[WARN] next-day fetch failed: {e}")
            df_next = pd.DataFrame(columns=df.columns)
    else:
        df_next = df[df["date"] == next_date] if next_date is not None else pd.DataFrame(columns=df.columns)

    tr = df[df["date"] >= market_date - pd.DateOffset(months=TRAIN_MONTHS)]
    te = df[df["date"] == market_date]

    if tr.empty or te.empty:
        print("⚠ データ不足: train/test empty")
        return

    # maps for fast access
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

    if DEBUG:
        print(f"[DEBUG] market_date={market_date}, next_date={next_date}, positions_before={len(old_positions)}")

    # ---------- 売却処理（満了）: 優先 next_open -> te_close -> last_available, apply SLIPPAGE & fees ----------
    for pos in old_positions:
        pos["held_days"] = pos.get("held_days", 0) + 1
        if DEBUG:
            print(f"[DEBUG] checking pos {pos['ticker']} held_days={pos['held_days']} hold_days={pos['hold_days']}")
        if pos["held_days"] >= pos["hold_days"]:
            sell_price = None
            sell_source = None
            # 1) next_open
            if next_date is not None and pos["ticker"] in next_open_map:
                sell_price = next_open_map[pos["ticker"]] * (1.0 - SLIPPAGE)
                sell_source = f"next_open({next_date.date()})"
            # 2) today's close
            elif pos["ticker"] in te_close_map:
                sell_price = te_close_map[pos["ticker"]] * (1.0 - SLIPPAGE)
                sell_source = f"te_close({market_date.date()})"
            else:
                last_price = get_last_available_price(df, pos["ticker"], market_date)
                if last_price is not None:
                    sell_price = last_price * (1.0 - SLIPPAGE)
                    sell_source = "last_available_before_or_on_market_date"
                else:
                    sell_price = None
                    sell_source = "no_price_found"

            if sell_price is None:
                # Can't sell — keep position and log reason
                remaining.append(pos)
                sells_skipped_for_log.append(f"{pos['ticker']}:skip_no_price")
                if DEBUG:
                    print(f"[DEBUG] sell skipped for {pos['ticker']} reason=no_price")
                continue

            # compute shares, proceeds, subtract fees
            shares = pos["amount"] / pos["buy_price"] if pos["buy_price"] != 0 else 0.0
            proceeds = shares * sell_price
            buy_comm = COMMISSION * pos["amount"]
            sell_comm = COMMISSION * proceeds
            profit = proceeds - pos["amount"] - buy_comm - sell_comm

            realized_pnl += profit
            sold_lines.append(f"{pos['ticker']} ({pos['hold_days']}日): {((sell_price/pos['buy_price']-1)*100):+.2f}% ({profit:+,.0f}円) [{sell_source}]")
            sells_for_log.append(f"{pos['ticker']}:profit={profit:+.0f}:src={sell_source}")
        else:
            remaining.append(pos)

    # integrate skipped logs
    if sells_skipped_for_log:
        sells_for_log.extend(sells_skipped_for_log)

    capital += realized_pnl

    # ---------- 新規購入（シグナル:優先 next_open -> te_close -> last_available） ----------
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
                    if DEBUG:
                        print(f"[DEBUG] skipping buy {row.ticker} - no price available")
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

    # unrealized valuation (use market_date close as conservative valuation)
    unrealized = 0.0
    for pos in all_positions:
        cur_price = te_close_map.get(pos["ticker"], pos["buy_price"])
        unrealized += (cur_price / pos["buy_price"] - 1.0) * pos["amount"]

    current_equity = capital + unrealized

    # 状態保存
    state = {"capital": capital, "positions": all_positions}
    save_state(state)

    # ==== 売却ログ強化：sold_lines が空でもスキップ理由を表示する ====
    if not sold_lines and sells_for_log:
        sold_lines = [f"（売却処理は実行されたが全てスキップ: {', '.join(sells_for_log)}）"]
    elif not sold_lines:
        sold_lines = ["（売却なし）"]

    sold_str = "\n".join(sold_lines)
    sells_for_log_summary = ";".join(sells_for_log) if sells_for_log else ""
    buy_str = "\n".join([f"{p['ticker']} ({p['hold_days']}日保有) @ {p['buy_price']:.2f}" for p in new_positions]) if new_positions else "（新規購入なし）"

    run_date_jst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).date()
    msg = (
        f"📅 **{run_date_jst} トレード結果**\n"
        f"**市場データ日:** {market_date.date()}\n"
        f"**約定想定日（next trading date）:** {next_date.date() if next_date is not None else 'N/A'}\n"
        f"**売却:**\n{sold_str}\n"
        f"**売却(ログ簡易):** {sells_for_log_summary}\n"
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
                     sells=[sells_for_log_summary] if sells_for_log_summary else [])

# =========================
# メイン（JSTの土日スキップ）
# =========================
def main():
    weekday = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).weekday()
    if weekday >= 5:
        msg = f"🛌 {datetime.date.today()} は休場日のためスキップ"
        print(msg)
        notify_discord(msg)
        return

    try:
        all_df = load_all_data_fast(ALL_TICKERS)
    except Exception as e:
        print(f"[ERROR] data load failed: {e}")
        notify_discord(f"[ERROR] data load failed: {e}")
        return

    simulate_continuous(all_df)

if __name__ == "__main__":
    main()

















