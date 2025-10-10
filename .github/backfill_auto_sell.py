# backfill_auto_sell.py
# -*- coding: utf-8 -*-
"""
Backfill with:
 - AUTO_SELL_ON_EXPIRY = True
 - robust price fetch (yf.Ticker.history + yf.download, retries)
 - skip buys when price not available
 - duplicate-buy aggregation (combine)
 - MAX_TOTAL_EXPOSURE_FRAC to cap total invested amount
 - DEDUCT_ON_BUY toggle to control immediate cash deduction
Run: python backfill_auto_sell.py
"""
import os, json, shutil, datetime, traceback, time
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

# --------------------
# Debug / Config
# --------------------
DEBUG = False  # True にすると詳細ログ
TZ = datetime.timezone(datetime.timedelta(hours=9))
STATE_FILE = Path("equity_state.json")
LOG_FILE = Path("equity_log.csv")
BACKUP_DIR = Path("backups")
BACKUP_DIR.mkdir(exist_ok=True)

# Capital & risk params
INIT_CAPITAL = 1_000_000
PER_POS_FRAC = 0.20     # fraction of capital to allocate when buying (then divided across buys)
MAX_TOTAL_EXPOSURE_FRAC = 0.60  # <-- New: total invested (sum of positions.amount) max fraction of INIT_CAPITAL
COMMISSION = 0.0005
SLIPPAGE = 0.0002  # 0.02% slippage applied
DEDUCT_ON_BUY = True  # If False, buying doesn't immediately reduce 'capital' (useful for accounting choices)
AUTO_SELL_ON_EXPIRY = True

# --------------------
# Utilities
# --------------------
def backup_file(p: Path):
    if p.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = BACKUP_DIR / f"{p.stem}_backup_{stamp}{p.suffix}"
        shutil.copy2(p, dest)
        print(f"[backup] {p} -> {dest}")

def safe_load_state():
    if not STATE_FILE.exists():
        return {"capital": INIT_CAPITAL, "positions": []}
    try:
        return json.load(open(STATE_FILE, "r", encoding="utf-8"))
    except Exception:
        return {"capital": INIT_CAPITAL, "positions": []}

def safe_save_state(state):
    tmp = STATE_FILE.with_suffix(".tmp.json")
    json.dump(state, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    tmp.replace(STATE_FILE)

def append_log_row(row: dict):
    header = not LOG_FILE.exists()
    df = pd.DataFrame([row])
    df.to_csv(LOG_FILE, mode="a", index=False, header=header)

# --------------------
# Robust price fetch
# --------------------
def fetch_prices_window(ticker: str, start_date: datetime.date, end_date: datetime.date, retries=2, sleep_sec=0.8):
    for attempt in range(retries + 1):
        try:
            tk = yf.Ticker(ticker)
            df = None
            try:
                df = tk.history(start=start_date.isoformat(),
                                end=(end_date + datetime.timedelta(days=1)).isoformat(),
                                auto_adjust=False)
            except Exception:
                df = None
            if df is None or df.empty:
                try:
                    df = yf.download(ticker,
                                     start=start_date.isoformat(),
                                     end=(end_date + datetime.timedelta(days=1)).isoformat(),
                                     progress=False, auto_adjust=False)
                except Exception:
                    df = None
            if df is None or df.empty:
                if DEBUG:
                    print(f"[fetch] attempt {attempt} no data for {ticker} {start_date}~{end_date}")
                if attempt < retries:
                    time.sleep(sleep_sec)
                    continue
                return None
            df.index = pd.to_datetime(df.index).date
            # normalize column names
            rename_map = {}
            lower_cols = {c.lower(): c for c in df.columns}
            for cand in ["open","high","low","close","volume","adjclose"]:
                if cand in lower_cols:
                    rename_map[lower_cols[cand]] = cand.capitalize() if cand!="adjclose" else "AdjClose"
            if rename_map:
                df = df.rename(columns=rename_map)
            return df
        except Exception as e:
            if DEBUG:
                print(f"[WARN] fetch_prices_window attempt {attempt} failed for {ticker}: {e}")
            if attempt < retries:
                time.sleep(sleep_sec)
                continue
            return None

def get_execution_price_for_sell(ticker: str, market_date: datetime.date, lookahead_days=5):
    start = market_date - datetime.timedelta(days=7)
    end = market_date + datetime.timedelta(days=lookahead_days)
    df = fetch_prices_window(ticker, start, end)
    if df is None:
        return None, "no_price_data"
    try:
        next_dates = sorted([d for d in df.index if d > market_date])
    except Exception:
        next_dates = []
    if next_dates:
        nd = next_dates[0]
        if "Open" in df.columns and nd in df.index:
            val = df.at[nd, "Open"]
            if pd.notna(val):
                try:
                    return float(val), f"next_open({nd})"
                except Exception:
                    pass
    if market_date in df.index and "Close" in df.columns:
        val = df.at[market_date, "Close"]
        if pd.notna(val):
            try:
                return float(val), f"market_close({market_date})"
            except Exception:
                pass
    past = sorted([d for d in df.index if d <= market_date], reverse=True)
    if past:
        pd0 = past[0]
        if "Close" in df.columns:
            val = df.at[pd0, "Close"]
            if pd.notna(val):
                try:
                    return float(val), f"last_avail({pd0})"
                except Exception:
                    pass
    return None, "no_price_found"

def get_execution_price_for_buy(ticker: str, market_date: datetime.date, lookahead_days=5):
    return get_execution_price_for_sell(ticker, market_date, lookahead_days)

# --------------------
# Combine existing position with new buy
# --------------------
def combine_position(existing_pos: dict, buy_amount: float, buy_price_per_share: float, new_hold_days: int):
    old_amount = float(existing_pos.get("amount", 0.0))
    old_price = float(existing_pos.get("buy_price", 1.0))
    old_shares = old_amount / old_price if old_price != 0 else 0.0
    new_shares = buy_amount / buy_price_per_share if buy_price_per_share != 0 else 0.0
    total_shares = old_shares + new_shares
    total_amount = old_amount + buy_amount
    if total_shares > 0:
        new_buy_price = total_amount / total_shares
    else:
        new_buy_price = existing_pos.get("buy_price", buy_price_per_share)
    existing_pos["buy_price"] = float(new_buy_price)
    existing_pos["amount"] = float(total_amount)
    existing_pos["hold_days"] = int(max(existing_pos.get("hold_days", 0), int(new_hold_days)))
    existing_pos["held_days"] = 0
    if DEBUG:
        print(f"[combine] {existing_pos['ticker']} combined -> amount={total_amount:.0f}, buy_price={new_buy_price:.4f}, shares={total_shares:.4f}")
    return existing_pos

# --------------------
# Helper: current total exposure (sum of positions.amount)
# --------------------
def current_total_exposure(positions):
    return sum([float(p.get("amount", 0.0)) for p in positions])

# --------------------
# User trade events (example from user)
# --------------------
trade_events = {
    "2025-10-03": {"buys":[{"ticker":"MA","hold_days":5},{"ticker":"PSX","hold_days":5},{"ticker":"XOM","hold_days":5}], "sells":[]},
    "2025-10-06": {"buys":[{"ticker":"META","hold_days":5},{"ticker":"PSX","hold_days":5},{"ticker":"TSLA","hold_days":5}], "sells":[]},
    "2025-10-07": {"buys":[{"ticker":"CRWD","hold_days":5},{"ticker":"ADBE","hold_days":5},{"ticker":"OXY","hold_days":5}], "sells":[]},
    "2025-10-08": {"buys":[{"ticker":"F","hold_days":5},{"ticker":"PLTR","hold_days":5},{"ticker":"DHR","hold_days":5}], "sells":[]},
    "2025-10-09": {"buys":[{"ticker":"BLK","hold_days":5},{"ticker":"LMT","hold_days":5},{"ticker":"F","hold_days":5}], "sells":[]},
}

# --------------------
# Main backfill with exposure cap & combine
# --------------------
def replay_day_by_day(events_dict):
    backup_file(STATE_FILE)
    backup_file(LOG_FILE)

    state = safe_load_state()
    capital = float(state.get("capital", INIT_CAPITAL))
    positions = state.get("positions", [])

    event_dates = sorted([datetime.date.fromisoformat(d) for d in events_dict.keys()])
    start_date = event_dates[0]
    end_date = event_dates[-1]

    if not LOG_FILE.exists():
        append_log_row({
            "run_date_jst": datetime.datetime.now(TZ).date().isoformat(),
            "market_date": "",
            "status": "init",
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "equity": float(capital),
            "capital": float(capital),
            "num_positions": len(positions),
            "positions": json.dumps(positions, ensure_ascii=False),
            "buys": "",
            "sells": "",
            "daily_return_pct": 0.0,
            "error": ""
        })

    day = start_date
    while day <= end_date:
        print(f"\n[replay] simulating trading day = {day.isoformat()}")
        realized_today = 0.0
        sells_log = []
        buys_log = []

        # increment held_days
        for p in positions:
            p["held_days"] = p.get("held_days", 0) + 1

        # AUTO-SELL expired positions
        if AUTO_SELL_ON_EXPIRY:
            remaining = []
            for p in positions:
                if p.get("held_days", 0) >= p.get("hold_days", 0):
                    exec_price, src = get_execution_price_for_sell(p["ticker"], day)
                    if exec_price is None:
                        sells_log.append(f"{p['ticker']}:skip_no_price")
                        if DEBUG:
                            print(f"[auto-sell] skip {p['ticker']} on {day} (no price)")
                        remaining.append(p)
                        continue
                    exec_price_adj = exec_price * (1.0 - SLIPPAGE)
                    shares = p["amount"] / p["buy_price"] if p["buy_price"] != 0 else 0.0
                    proceeds = shares * exec_price_adj
                    buy_comm = COMMISSION * p["amount"]
                    sell_comm = COMMISSION * proceeds
                    profit = proceeds - p["amount"] - buy_comm - sell_comm
                    realized_today += profit
                    sells_log.append(f"{p['ticker']}:profit={profit:+.0f}:src={src}")
                    if DEBUG:
                        print(f"[auto-sell] {p['ticker']} sold at {exec_price_adj:.2f} ({src}), profit {profit:+.0f}")
                else:
                    remaining.append(p)
            positions = remaining

        # explicit sells (preserve behavior)
        ev = events_dict.get(day.isoformat(), {})
        for s in ev.get("sells", []):
            tkr = s["ticker"]
            specified_price = s.get("price")
            idx = next((i for i,p in enumerate(positions) if p["ticker"]==tkr), None)
            if idx is None:
                sells_log.append(f"{tkr}:no_pos_found")
                continue
            pos = positions.pop(idx)
            if specified_price is not None:
                exec_price = float(specified_price) * (1.0 - SLIPPAGE)
                src = "specified_price"
            else:
                exec_price_raw, src = get_execution_price_for_sell(tkr, day)
                if exec_price_raw is None:
                    sells_log.append(f"{tkr}:skip_no_price")
                    positions.append(pos)
                    continue
                exec_price = exec_price_raw * (1.0 - SLIPPAGE)
            shares = pos["amount"] / pos["buy_price"] if pos["buy_price"] != 0 else 0.0
            proceeds = shares * exec_price
            buy_comm = COMMISSION * pos["amount"]
            sell_comm = COMMISSION * proceeds
            profit = proceeds - pos["amount"] - buy_comm - sell_comm
            realized_today += profit
            sells_log.append(f"{tkr}:profit={profit:+.0f}:src={src}")
            if DEBUG:
                print(f"[sell-explicit] {tkr} sold at {exec_price:.2f}, profit {profit:+.0f}")

        capital += realized_today

        # BUY processing with exposure cap
        buys = ev.get("buys", [])
        # collect candidate prices (skip those with no price)
        candidate_prices = []
        for b in buys:
            tkr = b["ticker"]
            specified_price = b.get("price")
            hold_days_cfg = int(b.get("hold_days", 5))
            if specified_price is not None:
                candidate_prices.append((tkr, float(specified_price), "specified_price", hold_days_cfg))
            else:
                px, src = get_execution_price_for_buy(tkr, day)
                if px is None:
                    if DEBUG:
                        print(f"[buy-scan] skip {tkr} on {day} - no price ({src})")
                    continue
                candidate_prices.append((tkr, float(px), src, hold_days_cfg))

        # compute allowed new exposure based on MAX_TOTAL_EXPOSURE_FRAC
        current_exposure = current_total_exposure(positions)
        max_allowed_total = INIT_CAPITAL * float(MAX_TOTAL_EXPOSURE_FRAC)
        available_for_new = max(0.0, max_allowed_total - current_exposure)
        if DEBUG:
            print(f"[exposure] current={current_exposure:.0f}, max_allowed_total={max_allowed_total:.0f}, available_for_new={available_for_new:.0f}")

        n_valid = len(candidate_prices)
        if n_valid == 0 or available_for_new <= 0:
            if DEBUG:
                print(f"[buy] no valid buys or no available exposure on {day} -> skipping buys")
        else:
            # desired_total = capital * PER_POS_FRAC
            desired_total = float(capital) * float(PER_POS_FRAC)
            # but cannot exceed available_for_new
            allocatable_total = min(desired_total, available_for_new)
            if allocatable_total <= 0:
                if DEBUG:
                    print(f"[buy] allocatable_total <= 0 on {day} -> skipping buys")
            else:
                # distribute allocatable_total across valid candidates equally
                per_trade = allocatable_total / n_valid
                if DEBUG:
                    print(f"[buy] day={day} valid={n_valid} desired={desired_total:.0f} allocatable={allocatable_total:.0f} per_trade={per_trade:.0f}")
                for (tkr, px_raw, src, hold_days_cfg) in candidate_prices:
                    buy_price_adj = float(px_raw) * (1.0 + SLIPPAGE)
                    amount = float(per_trade)
                    # if DEDUCT_ON_BUY True deduct immediately, else not
                    if DEDUCT_ON_BUY:
                        capital -= amount
                    # combine if exists
                    idx = next((i for i,p in enumerate(positions) if p["ticker"]==tkr), None)
                    if idx is not None:
                        positions[idx] = combine_position(positions[idx], amount, buy_price_adj, hold_days_cfg)
                        buys_log.append(f"{tkr}:combined:hold={positions[idx]['hold_days']}:price={positions[idx]['buy_price']:.2f}:amt={int(positions[idx]['amount'])}:src={src}")
                        if DEBUG:
                            print(f"[buy-combine] {tkr} combined -> amount={int(positions[idx]['amount'])}, price={positions[idx]['buy_price']:.2f}")
                    else:
                        positions.append({
                            "ticker": tkr,
                            "buy_price": float(buy_price_adj),
                            "amount": float(amount),
                            "hold_days": int(hold_days_cfg),
                            "held_days": 0
                        })
                        buys_log.append(f"{tkr}:hold={hold_days_cfg}:price={buy_price_adj:.2f}:amt={int(amount)}:src={src}")
                        if DEBUG:
                            print(f"[buy] {tkr} at {buy_price_adj:.2f} src={src} amt={int(amount)}")

        # compute unrealized based on market_date close if available
        unrealized = 0.0
        for p in positions:
            cur_price = None
            try:
                dfp = fetch_prices_window(p["ticker"], day - datetime.timedelta(days=3), day + datetime.timedelta(days=1))
                if dfp is not None and not dfp.empty:
                    if day in dfp.index and "Close" in dfp.columns and not pd.isna(dfp.loc[day, "Close"]):
                        cur_price = float(dfp.loc[day, "Close"])
                    else:
                        past = sorted([d for d in dfp.index if d <= day], reverse=True)
                        if past:
                            cur_price = float(dfp.loc[past[0], "Close"])
            except Exception:
                cur_price = None
            if cur_price is None:
                cur_price = float(p["buy_price"])
            unrealized += (cur_price / float(p["buy_price"]) - 1.0) * float(p["amount"])

        equity = float(capital) + float(unrealized)

        # append log row
        prev_equity = None
        try:
            if LOG_FILE.exists():
                prev_df = pd.read_csv(LOG_FILE)
                if len(prev_df) > 0:
                    prev_equity = float(prev_df.iloc[-1]["equity"])
        except Exception:
            prev_equity = None
        daily_ret = 0.0
        if prev_equity is not None and prev_equity != 0:
            daily_ret = (equity / prev_equity - 1.0) * 100.0

        row = {
            "run_date_jst": datetime.datetime.now(TZ).date().isoformat(),
            "market_date": day.isoformat(),
            "status": "backfill",
            "realized_pnl": float(realized_today),
            "unrealized_pnl": float(unrealized),
            "equity": float(equity),
            "capital": float(capital),
            "num_positions": len(positions),
            "positions": ";".join([f"{p['ticker']}:{p['hold_days']}/{p.get('held_days',0)}:{int(p['amount'])}" for p in positions]),
            "buys": ";".join(buys_log),
            "sells": ";".join(sells_log),
            "daily_return_pct": float(daily_ret),
            "error": ""
        }
        append_log_row(row)

        # save state
        state = {"capital": float(capital), "positions": positions}
        safe_save_state(state)
        print(f"[done] {day} -> capital={int(capital)} equity={int(equity)} positions={len(positions)}")

        day = day + datetime.timedelta(days=1)
        time.sleep(0.5)

    print("\n[ALL DONE] Backfill complete. Backups in:", BACKUP_DIR)

# --------------------
# Run
# --------------------
if __name__ == "__main__":
    try:
        replay_day_by_day(trade_events)
    except Exception as e:
        print("[FATAL] exception during backfill:", e)
        traceback.print_exc()
