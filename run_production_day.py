#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_production_day.py
- 翌寄付き約定（next_open優先） / 売却・買付フラグ厳密化
- 買付時に即時で現金を差し引き（資金管理の厳格化）
- 日次CSVログ + 詳細JSONサマリ出力
- Discord通知（環境変数 DISCORD_WEBHOOK_URL）
- 土日スキップ（JST）
- 簡易保護: MAX_TOTAL_EXPOSURE_FRAC / MAX_PER_TICKER_FRAC を導入
"""

import os, sys, json, datetime, traceback, shutil, math
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import requests

# =========================
# 設定（変更可）
# =========================
START = "2015-01-01"
END = None
RANDOM_SEED = 42
TRAIN_MONTHS = 6
TOP_K = 3

# 資金管理
INIT_CAPITAL = 1_000_000
PER_POS_FRAC = 0.20      # 一度のトレードで同時に使う総資金割合（買い候補群合算に対する割合）
COMMISSION = 0.0005
SLIPPAGE = 0.0002        # 約定スリッページ
FIXED_SCORE_QUANTILE = 0.50
SOFTMAX_TEMP = 0.15
RET_CLIP_LOW, RET_CLIP_HIGH = -0.08, 0.30

# エクスポージャ制限（本番安全装置）
MAX_TOTAL_EXPOSURE_FRAC = float(os.getenv("MAX_TOTAL_EXPOSURE_FRAC", "0.6"))   # 合計割当 <= init_cap * this
MAX_PER_TICKER_FRAC    = float(os.getenv("MAX_PER_TICKER_FRAC", "0.25"))       # 1銘柄あたり上限

# ログ / ファイル
STATE_FILE = Path(os.getenv("STATE_FILE", "equity_state.json"))
LOG_FILE   = Path(os.getenv("LOG_FILE", "equity_log.csv"))
SUMMARY_DIR = Path(os.getenv("SUMMARY_DIR", "run_summaries"))
BACKUP_DIR = Path(os.getenv("BACKUP_DIR", "backups"))
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Discord
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL","").strip()

# 動作フラグ
ENABLE_CSV_LOG = True
DRY_RUN = False   # True にすると state 保存や Discord 通知は行わない（検証用）

# Universe (必要に応じて)
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
# ヘルパー関数
# =========================
def now_jst_date():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).date()

def backup_file(p: Path):
    if not p.exists():
        return None
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"{p.stem}_backup_{ts}{p.suffix}"
    shutil.copy2(p, dst)
    return dst

def notify_discord(content: str):
    # 簡素な通知（長文OK） - 環境変数で webhook 設定
    if not DISCORD_WEBHOOK_URL:
        print("[discord] webhook not configured, printing message instead:")
        print(content)
        return False
    try:
        r = requests.post(DISCORD_WEBHOOK_URL, json={"content": content}, timeout=10)
        return 200 <= r.status_code < 300
    except Exception as e:
        print(f"[discord] failed: {e}")
        return False

def append_csv_log(row: dict):
    if not ENABLE_CSV_LOG:
        return
    df = pd.DataFrame([row])
    header = not LOG_FILE.exists()
    df.to_csv(LOG_FILE, mode="a", index=False, header=header, encoding="utf-8")

def save_summary_json(summary: dict, run_date: datetime.date):
    fn = SUMMARY_DIR / f"summary_{run_date.isoformat()}.json"
    with fn.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

# === state helpers (applied_runs で冪等) ===
def load_state():
    if not STATE_FILE.exists():
        return {"capital": INIT_CAPITAL, "positions": [], "applied_runs": []}
    try:
        return json.load(STATE_FILE.open("r", encoding="utf-8"))
    except Exception:
        return {"capital": INIT_CAPITAL, "positions": [], "applied_runs": []}

def save_state(state: dict):
    # backup first
    backup_file(STATE_FILE)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

# =========================
# 特徴量・モデル（簡易、既存ロジック踏襲）
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

def load_all_data(tickers):
    print(f"[data] downloading {len(tickers)} tickers...")
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, group_by='ticker', threads=True, progress=False)
    out = []
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t not in raw.columns.levels[0]:
                    continue
                df = raw[t].copy().reset_index()
            else:
                df = raw.copy().reset_index()
            df.columns = [c.lower() for c in df.columns]
            need = {"date","open","high","low","close","volume"}
            if not need.issubset(set(df.columns)):
                continue
            feat = make_features(df, t)
            out.append(feat)
        except Exception as e:
            continue
    if not out:
        raise RuntimeError("no market data")
    all_df = pd.concat(out, ignore_index=True).sort_values(["date","ticker"]).reset_index(drop=True)
    print(f"[data] rows={len(all_df)} period={all_df['date'].min().date()} ~ {all_df['date'].max().date()}")
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
        mu[k] = float(pos.mean()) if len(pos)>0 else 0.0
    return mu

# =========================
# 約定価格取得ヘルパー
# =========================
def get_last_available_price(all_df, ticker, date):
    sub = all_df[(all_df["ticker"]==ticker) & (all_df["date"]<=date)].sort_values("date", ascending=False)
    if not sub.empty:
        return float(sub.iloc[0]["close"])
    return None

# =========================
# コア：1日分の実行（売却→買付） - 購入で即時資金差引、フラグ厳格化、ログ化
# =========================
def run_for_market_date(all_df):
    all_df = all_df.copy()
    all_df["date"] = pd.to_datetime(all_df["date"])
    market_date = all_df["date"].max()
    unique_dates = sorted(all_df["date"].unique())

    # next trading date
    try:
        idx = unique_dates.index(pd.Timestamp(market_date))
        next_date = unique_dates[idx+1] if idx+1 < len(unique_dates) else None
    except ValueError:
        next_date = None

    # try to fetch next-day via yfinance if missing (as in original)
    df_next = pd.DataFrame(columns=all_df.columns)
    if next_date is None:
        try:
            extra_start = (pd.to_datetime(market_date) + pd.Timedelta(days=1)).date().isoformat()
            extra_end   = (pd.to_datetime(market_date) + pd.Timedelta(days=4)).date().isoformat()
            extra_raw = yf.download(ALL_TICKERS, start=extra_start, end=extra_end, auto_adjust=True, group_by='ticker', threads=True, progress=False)
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
        except Exception:
            df_next = pd.DataFrame(columns=all_df.columns)
    else:
        df_next = all_df[all_df["date"]==next_date] if next_date is not None else pd.DataFrame(columns=all_df.columns)

    # train / test
    tr = all_df[all_df["date"] >= market_date - pd.DateOffset(months=TRAIN_MONTHS)]
    te = all_df[all_df["date"] == market_date]

    if tr.empty or te.empty:
        raise RuntimeError("train or test empty")

    # build quick maps
    te_close_map = {r.ticker: float(r.close) for r in te.itertuples()}
    te_open_map  = {r.ticker: float(r.open)  for r in te.itertuples()}
    next_open_map = {r.ticker: float(r.open) for r in df_next.itertuples()} if not df_next.empty else {}
    next_close_map = {r.ticker: float(r.close) for r in df_next.itertuples()} if not df_next.empty else {}

    # load state
    state = load_state()
    capital = float(state.get("capital", INIT_CAPITAL))
    positions = state.get("positions", [])
    applied_runs = set(state.get("applied_runs", []))

    realized_pnl = 0.0
    sells_log = []
    sells_skipped_log = []
    sells_events = []
    buys_log = []
    buys_events = []

    # ---------- SELL: held_days インクリメント -> 保有満了を売却 ----------
    remaining_positions = []
    for pos in positions:
        pos["held_days"] = int(pos.get("held_days", 0)) + 1
        # sell only if hold_days is reached
        if pos.get("hold_days") is not None and pos["held_days"] >= pos["hold_days"]:
            # determine sell price: next_open -> te_close -> last_available
            sell_price = None
            sell_src = None
            if next_date is not None and pos["ticker"] in next_open_map:
                sell_price = next_open_map[pos["ticker"]] * (1.0 - SLIPPAGE)
                sell_src = f"next_open({next_date.date()})"
            elif pos["ticker"] in te_close_map:
                sell_price = te_close_map[pos["ticker"]] * (1.0 - SLIPPAGE)
                sell_src = f"te_close({market_date.date()})"
            else:
                last_price = get_last_available_price(all_df, pos["ticker"], market_date)
                if last_price is not None:
                    sell_price = last_price * (1.0 - SLIPPAGE)
                    sell_src = "last_available"
                else:
                    sell_price = None
                    sell_src = "no_price"

            if sell_price is None:
                sells_skipped_log.append({"ticker": pos["ticker"], "reason": "no_price"})
                remaining_positions.append(pos)  # keep if can't sell
                continue

            # compute proceeds and profit
            buy_amount = float(pos.get("amount", 0.0))
            buy_price  = float(pos.get("buy_price", 0.0))
            shares = buy_amount / (buy_price + 1e-12) if buy_price>0 else 0.0
            proceeds = shares * sell_price
            buy_comm = COMMISSION * buy_amount
            sell_comm = COMMISSION * proceeds
            profit = proceeds - buy_amount - buy_comm - sell_comm

            realized_pnl += profit
            capital += proceeds - sell_comm  # add proceeds minus commission
            # Note: buy_comm was already accounted at time of buy (we will subtract at buy)
            sells_log.append({
                "ticker": pos["ticker"],
                "buy_amount": buy_amount,
                "buy_price": buy_price,
                "sell_price": float(sell_price),
                "proceeds": float(proceeds),
                "profit": float(profit),
                "src": sell_src
            })
            sells_events.append(f"{pos['ticker']}:profit={profit:+.0f}:src={sell_src}")
            # do not append to remaining (effectively removed)
        else:
            remaining_positions.append(pos)

    # ---------- BUY: シグナル算出、候補選定 ----------

    add_future_labels_inplace(tr)
    add_future_labels_inplace(te)
    models = train_models(tr)
    mu = estimate_mu(tr)
    preds = predict_models(models, te)

    scores = np.vstack([preds["p1"]*mu["1"], preds["p3"]*mu["3"], preds["p5"]*mu["5"]]).T
    best_score = scores.max(axis=1)
    te = te.copy()
    te["score"] = best_score
    thr = float(np.quantile(best_score, FIXED_SCORE_QUANTILE))
    candidate_df = te[te["score"] >= thr].sort_values("score", ascending=False).head(TOP_K)

    # combine duplicates if any (shouldn't normally happen)
    cand_list = []
    for r in candidate_df.itertuples():
        cand_list.append({"ticker": r.ticker, "score": r.score, "p1": preds["p1"][candidate_df.index.get_loc(r.Index)],
                         "p3": preds["p3"][candidate_df.index.get_loc(r.Index)],
                         "p5": preds["p5"][candidate_df.index.get_loc(r.Index)]})

    # gating: exposure caps
    total_current_exposure = sum([float(p.get("amount",0.0)) for p in remaining_positions])
    max_total_allowed = MAX_TOTAL_EXPOSURE_FRAC * INIT_CAPITAL

    # per ticker current exposure map
    cur_per_ticker = defaultdict(float)
    for p in remaining_positions:
        cur_per_ticker[p["ticker"]] += float(p.get("amount",0.0))
    per_ticker_cap = MAX_PER_TICKER_FRAC * INIT_CAPITAL

    # allocate sequentially: allocate budget = capital * PER_POS_FRAC (this is the 'available allocation' for this run)
    # but enforce that sum(new_allocations) + total_current_exposure <= max_total_allowed
    available_allocation = capital * PER_POS_FRAC
    # safety: also cap by remaining of max_total_allowed
    available_allocation = min(available_allocation, max(0.0, max_total_allowed - total_current_exposure))

    # If no allocation possible, skip buys
    remaining_allocation = available_allocation
    new_positions = []
    buys_skipped = []

    # Precompute ideal per-ticker (equal weight among candidates)
    num_cands = len(cand_list)
    if num_cands > 0:
        ideal_each = remaining_allocation / num_cands if num_cands>0 else 0.0
    else:
        ideal_each = 0.0

    # iterate candidates and apply strict checks (capital availability, per-ticker cap)
    for cand in cand_list:
        t = cand["ticker"]
        # compute per-candidate intended allocation (equal split)
        intended = ideal_each
        # but if cur_per_ticker[t] already near cap, skip or reduce
        max_for_ticker = max(0.0, per_ticker_cap - cur_per_ticker.get(t, 0.0))
        alloc = min(intended, remaining_allocation, max_for_ticker)
        # require a minimum allocation threshold to avoid tiny lot
        MIN_ALLOC = 1.0  # minimum yen allocation
        if alloc < MIN_ALLOC or remaining_allocation <= 0.0:
            buys_skipped.append({"ticker": t, "reason": "insufficient_allocation_or_per_ticker_cap"})
            continue

        # get buy price (next_open -> te_close -> last_available)
        buy_price = None
        buy_src = None
        if next_date is not None and t in next_open_map:
            buy_price = next_open_map[t] * (1.0 + SLIPPAGE)
            buy_src = f"next_open({next_date.date()})"
        elif t in te_close_map:
            buy_price = te_close_map[t] * (1.0 + SLIPPAGE)
            buy_src = f"te_close({market_date.date()})"
        else:
            last_price = get_last_available_price(all_df, t, market_date)
            if last_price is not None:
                buy_price = last_price * (1.0 + SLIPPAGE)
                buy_src = "last_available"
            else:
                buys_skipped.append({"ticker": t, "reason": "no_price"})
                continue

        # account for buy commission (we will subtract commission from capital at buy time)
        buy_comm = COMMISSION * alloc
        total_cost = alloc + buy_comm

        # double-check available capital (we deduct immediately)
        if total_cost > capital:
            # not enough cash now - try to reduce allocation to available - buy_comm
            possible_alloc = max(0.0, capital - buy_comm)
            if possible_alloc < MIN_ALLOC:
                buys_skipped.append({"ticker": t, "reason": "not_enough_capital"})
                continue
            alloc = min(alloc, possible_alloc)
            buy_comm = COMMISSION * alloc
            total_cost = alloc + buy_comm

        # perform buy: deduct capital immediately and create position entry
        # if existing position in remaining_positions, aggregate (weighted avg)
        found = None
        for pos in remaining_positions:
            if pos["ticker"] == t:
                found = pos; break

        if found:
            old_amt = float(found.get("amount", 0.0))
            old_price = float(found.get("buy_price", 0.0)) if old_amt>0 else buy_price
            new_amt = old_amt + alloc
            # weighted avg price
            new_price = (old_amt * old_price + alloc * buy_price) / (new_amt + 1e-12)
            found["amount"] = float(new_amt)
            found["buy_price"] = float(new_price)
            found["held_days"] = 0
            # keep hold_days as existing if present, otherwise set to candidate's chosen hold
            # choose hold based on best horizon
            best_idx = np.argmax([cand.get("p1",0.0)*mu["1"], cand.get("p3",0.0)*mu["3"], cand.get("p5",0.0)*mu["5"]])
            found["hold_days"] = int([1,3,5][best_idx])
            cur_per_ticker[t] += alloc
            buys_log.append({"ticker": t, "alloc": alloc, "buy_price": buy_price, "src": buy_src, "merged": True})
        else:
            best_idx = np.argmax([cand.get("p1",0.0)*mu["1"], cand.get("p3",0.0)*mu["3"], cand.get("p5",0.0)*mu["5"]])
            hold_days = int([1,3,5][best_idx])
            new_pos = {
                "ticker": t,
                "buy_price": float(buy_price),
                "amount": float(alloc),
                "hold_days": int(hold_days),
                "held_days": 0
            }
            remaining_positions.append(new_pos)
            cur_per_ticker[t] += alloc
            buys_log.append({"ticker": t, "alloc": alloc, "buy_price": buy_price, "src": buy_src, "merged": False})

        # deduct immediately
        capital -= float(total_cost)
        remaining_allocation -= alloc

        buys_events.append(f"{t}:hold={int([1,3,5][best_idx])}:price={buy_price:.2f}:amt={alloc:.0f}:src={buy_src}")

    # Finalize state
    # compute unrealized PnL using conservative te_close prices (if missing use buy price)
    unrealized = 0.0
    final_positions = remaining_positions
    for p in final_positions:
        cur_price = te_close_map.get(p["ticker"], p["buy_price"])
        unrealized += (cur_price / (p["buy_price"] + 1e-12) - 1.0) * float(p["amount"])

    equity = capital + unrealized

    # update and persist state
    run_date = now_jst_date()
    run_id = f"{run_date.isoformat()}|{market_date.date().isoformat()}"
    if run_id not in applied_runs:
        applied_runs.add(run_id)
    state_out = {
        "capital": float(capital),
        "positions": final_positions,
        "applied_runs": list(applied_runs)
    }
    if not DRY_RUN:
        save_state(state_out)

    # build human summary
    sold_lines_human = []
    for s in sells_log:
        sold_lines_human.append(f"{s['ticker']} sold at {s['sell_price']:.2f} profit={s['profit']:+.0f} src={s['src']}")
    if not sold_lines_human and sells_skipped_log:
        sold_lines_human = [f"sell skipped: {','.join([sk['ticker'] for sk in sells_skipped_log])}"]

    buy_lines_human = []
    for b in buys_log:
        buy_lines_human.append(f"{b['ticker']} buy amt={b['alloc']:.0f} @ {b['buy_price']:.2f} src={b['src']}")

    msg = (
        f"📅 **{run_date.isoformat()} トレード結果**\n"
        f"**市場データ日:** {market_date.date().isoformat()}\n"
        f"**約定想定日:** {next_date.date().isoformat() if next_date is not None else 'N/A'}\n"
        f"**売却:**\n" + ("\n".join(sold_lines_human) if sold_lines_human else "（売却なし）") + "\n\n"
        f"**売却(ログ簡易):** {','.join(sells_events) if sells_events else ''}\n"
        f"**新規購入:**\n" + ("\n".join(buy_lines_human) if buy_lines_human else "（新規購入なし）") + "\n\n"
        f"**実現損益:** {realized_pnl:+,.0f}円\n"
        f"**含み損益:** {unrealized:+,.0f}円\n"
        f"**評価残高（含み込み）:** {equity:,.0f}円\n"
        f"**確定残高（現金）:** {capital:,.0f}円\n"
        f"**ポジション数:** {len(final_positions)}\n"
        f"**total_exposure:** {sum([p['amount'] for p in final_positions]):,.0f} / cap_limit={MAX_TOTAL_EXPOSURE_FRAC*INIT_CAPITAL:,.0f}\n"
    )

    # CSV row
    pos_summary = ";".join([f"{p['ticker']}:{int(p['amount'])}/{p.get('hold_days',0)}d" for p in final_positions])
    buys_field = ";".join(buys_events) if buys_events else ""
    sells_field = ";".join(sells_events) if sells_events else ""
    prev_equity = None
    if LOG_FILE.exists():
        try:
            prev = pd.read_csv(LOG_FILE)
            if len(prev)>0:
                prev_equity = float(prev.iloc[-1]["equity"])
        except Exception:
            prev_equity = None
    daily_return_pct = 0.0
    if prev_equity is not None and prev_equity != 0:
        daily_return_pct = (equity / prev_equity - 1.0) * 100.0

    csv_row = {
        "run_date_jst": run_date.isoformat(),
        "market_date": market_date.date().isoformat(),
        "status": "ok",
        "realized_pnl": float(realized_pnl),
        "unrealized_pnl": float(unrealized),
        "equity": float(equity),
        "capital": float(capital),
        "num_positions": len(final_positions),
        "positions": pos_summary,
        "buys": buys_field,
        "sells": sells_field,
        "daily_return_pct": float(daily_return_pct),
        "error": ""
    }

    # persist logs and summary json
    if not DRY_RUN:
        append_csv_log(csv_row)
        summary = {
            "run_date": run_date.isoformat(),
            "market_date": market_date.date().isoformat(),
            "status": "ok",
            "realized_pnl": float(realized_pnl),
            "unrealized_pnl": float(unrealized),
            "equity": float(equity),
            "capital": float(capital),
            "num_positions": len(final_positions),
            "positions": final_positions,
            "buys": buys_log,
            "sells": sells_log,
            "skipped_buys": buys_skipped,
            "skipped_sells": sells_skipped_log,
            "total_exposure": float(sum([p['amount'] for p in final_positions])),
            "flags": _collect_flags(final_positions)
        }
        save_summary_json(summary, run_date)

    # Discord notify
    notify_discord(msg)

    # return for testing
    return {"csv_row": csv_row, "summary": summary}

def _collect_flags(final_positions):
    flags = []
    total_exposure = sum([p["amount"] for p in final_positions])
    if total_exposure > MAX_TOTAL_EXPOSURE_FRAC * INIT_CAPITAL:
        flags.append(f"TOTAL_EXPOSURE_HIGH: {total_exposure/INIT_CAPITAL:.2f} > {MAX_TOTAL_EXPOSURE_FRAC}")
    # per ticker
    per = defaultdict(float)
    for p in final_positions:
        per[p["ticker"]] += p["amount"]
    for t,a in per.items():
        if a > MAX_PER_TICKER_FRAC * INIT_CAPITAL:
            flags.append(f"TICKER_CONCENTRATION_HIGH: {t}={a} > {MAX_PER_TICKER_FRAC*INIT_CAPITAL}")
    return flags

# =========================
# メイン（JST 土日スキップ）
# =========================
def main():
    # skip weekends (JST)
    wd = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).weekday()
    if wd >= 5:
        msg = f"🛌 {now_jst_date().isoformat()} は土日でスキップ"
        print(msg)
        notify_discord(msg)
        return

    try:
        backup_file(STATE_FILE)
        backup_file(LOG_FILE)
    except Exception as e:
        print(f"[warn] backup failed: {e}")

    try:
        all_df = load_all_data(ALL_TICKERS)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[error] data load failed: {e}\n{tb}")
        notify_discord(f"[ERROR] data load failed: {e}")
        return

    try:
        result = run_for_market_date(all_df)
        print("[info] run completed")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[error] run failed: {e}\n{tb}")
        notify_discord(f"[ERROR] run failed: {e}\n{tb}")
        return

if __name__ == "__main__":
    main()
















