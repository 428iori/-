#!/usr/bin/env python3
# diagnose_trading_state.py
# Usage:
#  python diagnose_trading_state.py --log equity_log.csv --state equity_state.json --init-capital 1000000

import argparse, json, csv, sys, re
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import datetime

POS_RE = re.compile(r"([^:;]+):([^:;]+):([^;]+)")  # ticker:hold/held:amount
BUY_AMT_RE = re.compile(r"amt=([0-9\.]+)")
PRICE_RE = re.compile(r"price=([0-9\.]+)")
SRC_RE = re.compile(r"src=([^;:]+)")
COMBINED_RE = re.compile(r"combined", re.IGNORECASE)
FALLBACK_KEYS = ["fallback", "fallback_1.0", "fallback_price"]

def parse_positions_field(s):
    """Parse positions field like 'PSX:5/0:140307;META:5/4:53333' -> list of dicts"""
    out = []
    if not s or pd.isna(s): 
        return out
    parts = [p for p in s.split(";") if p.strip()]
    for p in parts:
        # try ticker:hold/held:amount
        m = POS_RE.match(p.strip())
        if m:
            ticker = m.group(1).strip()
            hold = m.group(2).strip()
            amt = m.group(3).strip()
            try:
                amt_v = float(amt)
            except:
                amt_v = np.nan
            out.append({"ticker": ticker, "hold_spec": hold, "amount": amt_v, "raw": p.strip()})
        else:
            # fallback: try to salvage ticker and amount
            toks = p.strip().split(":")
            if toks:
                ticker = toks[0]
                # find last numeric token
                nums = [t for t in toks if re.match(r"^[0-9\.]+$", t)]
                amt_v = float(nums[-1]) if nums else np.nan
                out.append({"ticker": ticker, "hold_spec": None, "amount": amt_v, "raw": p.strip()})
    return out

def parse_buys_field(s):
    """Parse buys like 'OXY:hold=5:price=43.70:amt=66666:src=te_close(2025-10-09)'"""
    out = []
    if not s or pd.isna(s):
        return out
    parts = [p for p in s.split(",") if p.strip()]
    # sometimes semicolon-separated
    if len(parts)==1 and ";" in s:
        parts = [p for p in s.split(";") if p.strip()]
    for p in parts:
        p = p.strip()
        # ticker at start
        toks = p.split(":")
        ticker = toks[0] if toks else ""
        amt_m = BUY_AMT_RE.search(p)
        price_m = PRICE_RE.search(p)
        src_m = SRC_RE.search(p)
        amt = float(amt_m.group(1)) if amt_m else np.nan
        price = float(price_m.group(1)) if price_m else np.nan
        src = src_m.group(1) if src_m else ""
        out.append({"ticker": ticker, "amt": amt, "price": price, "src": src, "raw": p})
    return out

def flag_fallback_in_text(s):
    if not s or pd.isna(s):
        return False
    low = s.lower()
    for k in FALLBACK_KEYS:
        if k in low:
            return True
    return False

def summarize_log(df, init_capital):
    rows = []
    # accumulate metrics
    total_logs = len(df)
    fallback_buy_count = 0
    total_buy_events = 0
    total_sell_events = 0
    sells_count_lines = 0
    exposure_by_ticker = defaultdict(float)
    last_capital = None
    last_equity = None
    last_row = None
    combined_count = 0
    buy_sources = Counter()
    sell_sources = Counter()
    rows_no_trades = 0

    for idx, row in df.iterrows():
        last_row = row
        last_capital = safe_float(row.get("capital"))
        last_equity = safe_float(row.get("equity"))

        # positions
        pos_list = parse_positions_field(row.get("positions","") or "")
        for p in pos_list:
            if not np.isnan(p.get("amount", np.nan)):
                exposure_by_ticker[p["ticker"]] += float(p["amount"])

        # buys
        buys = row.get("buys","")
        buys_parsed = parse_buys_field(buys)
        if buys_parsed:
            for b in buys_parsed:
                total_buy_events += 1
                src = (b.get("src") or "").lower()
                buy_sources[src] += 1
                if any(k in src for k in FALLBACK_KEYS) or flag_fallback_in_text(b.get("raw","")):
                    fallback_buy_count += 1
                # also count combined marker
                if COMBINED_RE.search(b.get("raw","") or ""):
                    combined_count += 1
        else:
            rows_no_trades += 1

        # sells
        sells = row.get("sells","")
        if sells and not pd.isna(sells) and str(sells).strip():
            total_sell_events += 1
            sells_count_lines += 1
            # try to check source
            if "src=" in sells:
                m = SRC_RE.search(sells)
                if m:
                    sell_sources[m.group(1).lower()] += 1

    total_exposure = sum(exposure_by_ticker.values())
    exposure_frac = total_exposure / float(init_capital) if init_capital>0 else np.nan

    # per ticker top list
    tickers_sorted = sorted(exposure_by_ticker.items(), key=lambda x: x[1], reverse=True)
    top5 = tickers_sorted[:10]

    fallback_rate = (fallback_buy_count / total_buy_events) if total_buy_events>0 else 0.0
    sell_rate = (sells_count_lines / total_logs) if total_logs>0 else 0.0
    no_trade_pct = rows_no_trades / total_logs if total_logs>0 else 0.0

    # concentration flags
    flags = []
    if exposure_frac > 0.6:
        flags.append(f"TOTAL_EXPOSURE_HIGH: exposure_frac={exposure_frac:.2f} (>0.6)")
    elif exposure_frac > 0.4:
        flags.append(f"TOTAL_EXPOSURE_MEDIUM: exposure_frac={exposure_frac:.2f} (>0.4)")

    # ticker concentration
    if top5:
        top_ticker, top_amt = top5[0]
        if (top_amt / init_capital) > 0.25:
            flags.append(f"TICKER_CONCENTRATION_HIGH: {top_ticker}={top_amt:.0f} ({top_amt/init_capital:.2%})")
        elif (top_amt / init_capital) > 0.15:
            flags.append(f"TICKER_CONCENTRATION_MEDIUM: {top_ticker}={top_amt:.0f} ({top_amt/init_capital:.2%})")

    if fallback_rate > 0.10:
        flags.append(f"FALLBACK_RATE_HIGH: {fallback_rate:.2%} buys use fallback")
    if sell_rate < 0.05:
        flags.append(f"LOW_SELL_RATE: sells appear in only {sell_rate:.2%} of log rows")

    # last row summary
    last_summary = {}
    if last_row is not None:
        last_summary = {
            "run_date_jst": last_row.get("run_date_jst"),
            "market_date": last_row.get("market_date"),
            "status": last_row.get("status"),
            "realized_pnl": safe_float(last_row.get("realized_pnl")),
            "unrealized_pnl": safe_float(last_row.get("unrealized_pnl")),
            "equity": safe_float(last_row.get("equity")),
            "capital": safe_float(last_row.get("capital")),
            "num_positions": int_or_none(last_row.get("num_positions"))
        }

    diagnostics = {
        "total_log_rows": total_logs,
        "last_summary": last_summary,
        "total_exposure": total_exposure,
        "exposure_frac": exposure_frac,
        "top_positions": [{"ticker": t, "amount": a, "pct_of_init": a/init_capital} for t,a in top5],
        "num_buy_events": total_buy_events,
        "num_sell_event_rows": sells_count_lines,
        "fallback_buy_count": fallback_buy_count,
        "fallback_rate": fallback_rate,
        "sell_rate": sell_rate,
        "no_trade_pct": no_trade_pct,
        "buy_sources_top": buy_sources.most_common(10),
        "sell_sources_top": sell_sources.most_common(10),
        "combined_buy_lines": combined_count,
        "flags": flags,
        "timestamp": datetime.datetime.now().isoformat()
    }

    return diagnostics, exposure_by_ticker

def safe_float(x):
    try:
        return float(x) if x is not None and x!='' and not pd.isna(x) else np.nan
    except:
        return np.nan

def int_or_none(x):
    try:
        return int(x)
    except:
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, help="Path to equity_log.csv")
    p.add_argument("--state", required=False, help="Path to equity_state.json (optional)")
    p.add_argument("--init-capital", type=float, default=1_000_000, help="Initial capital (for exposure fraction)")
    p.add_argument("--out-json", default="diagnostics.json", help="Output diagnostics JSON")
    p.add_argument("--out-csv", default="diagnostics.csv", help="Output per-ticker CSV")
    args = p.parse_args()

    logp = Path(args.log)
    if not logp.exists():
        print("Log file not found:", logp, file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(logp, dtype=str).fillna("")
    # ensure expected columns exist
    expected_cols = ["run_date_jst","market_date","status","realized_pnl","unrealized_pnl","equity","capital","num_positions","positions","buys","sells","daily_return_pct","error"]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = ""

    diagnostics, exposure_by_ticker = summarize_log(df, args.init_capital)

    # print human summary
    print("=== Trading State Diagnostics ===")
    print(f"Rows analyzed: {diagnostics['total_log_rows']}")
    ls = diagnostics["last_summary"]
    if ls:
        print(f"Last run: run_date={ls.get('run_date_jst')} market_date={ls.get('market_date')} status={ls.get('status')}")
        print(f"Capital: {ls.get('capital')} | Equity: {ls.get('equity')} | unrealized: {ls.get('unrealized_pnl')}")
    print(f"Total exposure (sum of 'amount' in positions fields): {diagnostics['total_exposure']:.0f}")
    print(f"Exposure fraction of init capital: {diagnostics['exposure_frac']:.2%}")
    print(f"Top positions:")
    for it in diagnostics["top_positions"]:
        print(f"  - {it['ticker']}: {it['amount']:.0f} ({it['pct_of_init']:.2%} of init capital)")
    print(f"Buy events: {diagnostics['num_buy_events']}, fallback buys: {diagnostics['fallback_buy_count']} (rate {diagnostics['fallback_rate']:.2%})")
    print(f"Log rows with sells: {diagnostics['num_sell_event_rows']} ({diagnostics['sell_rate']:.2%})")
    if diagnostics["combined_buy_lines"]:
        print(f"Combined-buy markers found in buys: {diagnostics['combined_buy_lines']}")
    print("Flags:")
    if diagnostics["flags"]:
        for f in diagnostics["flags"]:
            print("  -", f)
    else:
        print("  - None")

    # write JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    # write CSV per-ticker exposure
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker","exposure","pct_of_init"])
        for t,a in sorted(exposure_by_ticker.items(), key=lambda x: x[1], reverse=True):
            w.writerow([t, f"{a:.2f}", f"{a/args.init_capital:.4f}"])

    print(f"\nDiagnostics written to {args.out_json} and {args.out_csv}")

if __name__ == "__main__":
    main()
