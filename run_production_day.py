# -*- coding: utf-8 -*-
"""
安全版フル実行スクリプト：
- 翌寄付き約定（next-open）優先、フォールバック（当日終値→直近available）
- 売却ログ強化（売却発生 / スキップ理由 を必ず通知＆CSVに出力）
- 既存の equity_state.json / equity_log.csv を尊重・初期化（無ければ作成）
- 例外時も CSV にエラーログを残す
- JST（Asia/Tokyo）基準、土日スキップ（ただしスキップ行はログに残す）
"""
import os, json, datetime, traceback, requests
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# -------------------
# Config
# -------------------
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
SLIPPAGE = 0.0002   # 0.02%

# Files / endpoints
LOG_FILE = Path("equity_log.csv")
STATE_FILE = Path("equity_state.json")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# Behavior
DEBUG = True

# Universe (adjust as needed)
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

TZ = datetime.timezone(datetime.timedelta(hours=9))  # JST

# -------------------
# Utilities: safe file ops & logging
# -------------------
def safe_write_json(path: Path, data):
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def ensure_files_exist():
    # state
    if not STATE_FILE.exists():
        init_state = {"capital": INIT_CAPITAL, "positions": []}
        safe_write_json(STATE_FILE, init_state)
        print(f"[INIT] created {STATE_FILE}")
    # log
    if not LOG_FILE.exists():
        header = {
            "run_date_jst": datetime.datetime.now(TZ).date().isoformat(),
            "market_date": "",
            "status": "init",
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "equity": float(INIT_CAPITAL),
            "capital": float(INIT_CAPITAL),
            "num_positions": 0,
            "positions": "",
            "buys": "",
            "sells": "",
            "daily_return_pct": 0.0,
            "error": ""
        }
        pd.DataFrame([header]).to_csv(LOG_FILE, index=False)
        print(f"[INIT] created {LOG_FILE}")

def append_daily_log(row: dict):
    try:
        header = not LOG_FILE.exists()
        pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", index=False, header=header)
    except Exception as e:
        # fallback robust append
        try:
            if LOG_FILE.exists():
                prev = pd.read_csv(LOG_FILE)
            else:
                prev = pd.DataFrame()
        except Exception:
            prev = pd.DataFrame()
        new = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
        new.to_csv(LOG_FILE, index=False)

def load_state():
    if not STATE_FILE.exists():
        return {"capital": INIT_CAPITAL, "positions": []}
    try:
        return json.load(open(STATE_FILE, "r", encoding="utf-8"))
    except Exception:
        return {"capital": INIT_CAPITAL, "positions": []}

def save_state(state):
    try:
        safe_write_json(STATE_FILE, state)
    except Exception as e:
        print("[WARN] failed to save state:", e)

def notify_discord(msg):
    if not DISCORD_WEBHOOK_URL:
        print(msg)
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg}, timeout=5)
    except Exception as e:
        print("[WARN] Discord notify failed:", e)
        print(msg)

# -------------------
# Features / model helpers
# -------------------
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
    print(f"[DATA] downloading {len(tickers)} tickers...")
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, group_by='ticker', threads=True, progress=False)
    out = []
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t not in raw.columns.levels[0]:
                    if DEBUG: print(f"[WARN] no data for {t}")
                    continue
                df = raw[t].reset_index()
            else:
                df = raw.reset_index()
            df.columns = [c.lower() for c in df.columns]
            need = {"date","open","high","low","close","volume"}
            if not need.issubset(set(df.columns)):
                if DEBUG: print(f"[WARN] missing cols for {t}")
                continue
            feat = make_features(df, t)
            out.append(feat)
        except Exception as e:
            if DEBUG: print("[ERR]", t, e)
            continue
    if not out:
        raise RuntimeError("No data fetched.")
    all_df = pd.concat(out, ignore_index=True).sort_values(["date","ticker"]).reset_index(drop=True)
    print(f"[DATA] rows={len(all_df)}  period: {all_df['date'].min().date()} ~ {all_df['date'].max().date()}")
    return all_df

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

def get_last_available_price(all_df_local, ticker, date):
    sub = all_df_local[(all_df_local["ticker"]==ticker) & (all_df_local["date"]<=date)].sort_values("date", ascending=False)
    if not sub.empty:
        return float(sub.iloc[0]["close"])
    return None

# -------------------
# Core: simulate with next-open + fallback + robust logging
# -------------------
def simulate_continuous(all_df):
    df = all_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    unique_dates = sorted(df["date"].unique())
    market_date = df["date"].max()

    # detect next trading date
    try:
        idx_today = unique_dates.index(pd.Timestamp(market_date))
        next_date = unique_dates[idx_today + 1] if idx_today + 1 < len(unique_dates) else None
    except ValueError:
        next_date = None

    # if next_date missing, attempt to fetch a few calendar days after market_date
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
            if DEBUG: print("[WARN] next-day fetch failed:", e)
            df_next = pd.DataFrame(columns=df.columns)
    else:
        df_next = df[df["date"] == next_date] if next_date is not None else pd.DataFrame(columns=df.columns)

    # prepare train/test
    tr = df[df["date"] >= market_date - pd.DateOffset(months=TRAIN_MONTHS)]
    te = df[df["date"] == market_date]
    if tr.empty or te.empty:
        raise RuntimeError("train/test empty")

    te_close_map = {r.ticker: float(r.close) for r in te.itertuples()}
    df_next_open_map = {r.ticker: float(r.open) for r in df_next.itertuples()} if not df_next.empty else {}
    df_next_close_map = {r.ticker: float(r.close) for r in df_next.itertuples()} if not df_next.empty else {}

    state = load_state()
    capital = state.get("capital", INIT_CAPITAL)
    old_positions = state.get("positions", [])
    realized_pnl = 0.0
    sold_lines = []
    sells_for_log = []
    sells_skipped_for_log = []
    remaining = []

    if DEBUG:
        print(f"[DEBUG] market_date={market_date}, next_date={next_date}, old_positions={len(old_positions)}")

    # SELL: check hold days; priority: next_open -> te_close -> last_available (apply slippage and fees)
    for pos in old_positions:
        pos["held_days"] = pos.get("held_days", 0) + 1
        if pos["held_days"] >= pos["hold_days"]:
            sell_price = None
            sell_source = None
            if next_date is not None and pos["ticker"] in df_next_open_map:
                sell_price = df_next_open_map[pos["ticker"]] * (1.0 - SLIPPAGE)
                sell_source = f"next_open({next_date.date()})"
            elif pos["ticker"] in te_close_map:
                sell_price = te_close_map[pos["ticker"]] * (1.0 - SLIPPAGE)
                sell_source = f"te_close({market_date.date()})"
            else:
                last = get_last_available_price(df, pos["ticker"], market_date)
                if last is not None:
                    sell_price = last * (1.0 - SLIPPAGE)
                    sell_source = "last_available_before_or_on_market_date"
                else:
                    sell_price = None
                    sell_source = "no_price_found"

            if sell_price is None:
                remaining.append(pos)
                sells_skipped_for_log.append(f"{pos['ticker']}:skip_no_price")
                if DEBUG: print(f"[DEBUG] sell skipped {pos['ticker']}")
                continue

            shares = pos["amount"] / pos["buy_price"] if pos["buy_price"] != 0 else 0.0
            proceeds = shares * sell_price
            buy_comm = COMMISSION * pos["amount"]
            sell_comm = COMMISSION * proceeds
            profit = proceeds - pos["amount"] - buy_comm - sell_comm

            realized_pnl += profit
            sold_lines.append(f"{pos['ticker']} ({pos['hold_days']}d): {((sell_price/pos['buy_price']-1)*100):+.2f}% ({profit:+,.0f}円) [{sell_source}]")
            sells_for_log.append(f"{pos['ticker']}:profit={profit:+.0f}:src={sell_source}")
        else:
            remaining.append(pos)

    # merge skipped
    if sells_skipped_for_log:
        sells_for_log.extend(sells_skipped_for_log)
    capital += realized_pnl

    # BUY: predict on te, pick top by score, determine buy_price by next_open preference
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
            if next_date is not None and row.ticker in df_next_open_map:
                buy_price = df_next_open_map[row.ticker] * (1.0 + SLIPPAGE)
                buy_source = f"next_open({next_date.date()})"
            elif row.ticker in te_close_map:
                buy_price = te_close_map[row.ticker] * (1.0 + SLIPPAGE)
                buy_source = f"te_close({market_date.date()})"
            else:
                lastp = get_last_available_price(df, row.ticker, market_date)
                if lastp is not None:
                    buy_price = lastp * (1.0 + SLIPPAGE)
                    buy_source = "last_available_before_or_on_market_date"
                else:
                    if DEBUG: print(f"[DEBUG] skipping buy {row.ticker} - no price")
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

    # unrealized valuation: use market_date close if available else buy_price
    unrealized = 0.0
    for pos in all_positions:
        cur_price = te_close_map.get(pos["ticker"], pos["buy_price"])
        unrealized += (cur_price / pos["buy_price"] - 1.0) * pos["amount"]

    current_equity = capital + unrealized

    # save state
    state = {"capital": capital, "positions": all_positions}
    save_state(state)

    # ensure sold_lines / sells_for_log visible even if empty
    if not sold_lines and sells_for_log:
        sold_lines = [f"（売却処理は実行されたが全てスキップ: {', '.join(sells_for_log)}）"]
    elif not sold_lines:
        sold_lines = ["（売却なし）"]

    sold_str = "\n".join(sold_lines)
    sells_for_log_summary = ";".join(sells_for_log) if sells_for_log else ""
    buy_str = "\n".join([f"{p['ticker']} ({p['hold_days']}d) @ {p['buy_price']:.2f}" for p in new_positions]) if new_positions else "（新規購入なし）"

    run_date_jst = datetime.datetime.now(TZ).date()
    msg = (
        f"📅 **{run_date_jst} トレード結果**\n"
        f"**市場データ日:** {market_date.date()}\n"
        f"**約定想定日:** {next_date.date() if next_date is not None else 'N/A'}\n"
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

    # CSV row
    prev_equity = None
    try:
        prev_df = pd.read_csv(LOG_FILE)
        if len(prev_df)>0:
            prev_equity = float(prev_df.iloc[-1]["equity"])
    except Exception:
        prev_equity = None
    daily_ret = 0.0
    if prev_equity is not None and prev_equity != 0:
        daily_ret = (current_equity / prev_equity - 1.0) * 100.0

    row = {
        "run_date_jst": run_date_jst.isoformat(),
        "market_date": pd.to_datetime(market_date).date().isoformat(),
        "status": "ok",
        "realized_pnl": float(realized_pnl),
        "unrealized_pnl": float(unrealized),
        "equity": float(current_equity),
        "capital": float(capital),
        "num_positions": len(all_positions),
        "positions": ";".join([f"{p['ticker']}:{p['hold_days']}/{p.get('held_days',0)}:{int(p['amount'])}" for p in all_positions]),
        "buys": ";".join(buys_for_log) if buys_for_log else "",
        "sells": sells_for_log_summary,
        "daily_return_pct": float(daily_ret),
        "error": ""
    }
    append_daily_log(row)

# -------------------
# Main wrapper with robust error logging
# -------------------
def main():
    ensure_files_exist()
    # skip weekend but still log skip row
    today = datetime.datetime.now(TZ).date()
    weekday = today.weekday()
    if weekday >= 5:
        row = {
            "run_date_jst": today.isoformat(),
            "market_date": "",
            "status": "weekend_skip",
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "equity": float(load_state().get("capital", INIT_CAPITAL)),
            "capital": float(load_state().get("capital", INIT_CAPITAL)),
            "num_positions": len(load_state().get("positions", [])),
            "positions": "",
            "buys": "",
            "sells": "",
            "daily_return_pct": 0.0,
            "error": ""
        }
        append_daily_log(row)
        print("[INFO] weekend - logged skip")
        return

    try:
        all_df = load_all_data_fast(ALL_TICKERS)
        simulate_continuous(all_df)
    except Exception as e:
        tb = traceback.format_exc()
        err_msg = str(e)[:2000]
        print("[ERROR] exception during run:", err_msg)
        # ensure we append an error row so files always show activity
        run_date = datetime.datetime.now(TZ).date()
        row = {
            "run_date_jst": run_date.isoformat(),
            "market_date": "",
            "status": "error",
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "equity": float(load_state().get("capital", INIT_CAPITAL)),
            "capital": float(load_state().get("capital", INIT_CAPITAL)),
            "num_positions": len(load_state().get("positions", [])),
            "positions": "",
            "buys": "",
            "sells": "",
            "daily_return_pct": 0.0,
            "error": err_msg + "\n" + tb[:4000]
        }
        append_daily_log(row)
        notify_discord(f"[ERROR] run failed: {err_msg}")

if __name__ == "__main__":
    main()

















