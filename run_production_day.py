# -*- coding: utf-8 -*-
"""
安全実行版：必ず equity_state.json と equity_log.csv を残すデバッグ重視スクリプト
- 既存ロジックの簡易代替＋堅牢なログ／エラーハンドリングを追加
- 実行ディレクトリにファイルを作成する（存在しなければ初期化）
- 土日スキップでもログ行を残す
"""
import os, json, datetime, traceback
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- 設定（必要なら調整） ----------
LOG_FILE = Path("equity_log.csv")
STATE_FILE = Path("equity_state.json")
TIMEZONE = datetime.timezone(datetime.timedelta(hours=9))  # JST

# 初期値
INIT_CAPITAL = 1_000_000

# ---------- ユーティリティ ----------
def ensure_files_exist():
    # state
    if not STATE_FILE.exists():
        init_state = {"capital": INIT_CAPITAL, "positions": []}
        safe_write_json(STATE_FILE, init_state)
        print(f"[INIT] created {STATE_FILE}")
    # log header
    if not LOG_FILE.exists():
        df = pd.DataFrame([{
            "run_date_jst": datetime.datetime.now(TIMEZONE).date().isoformat(),
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
        }])
        df.to_csv(LOG_FILE, index=False)
        print(f"[INIT] created {LOG_FILE} with header row")

def safe_write_json(path: Path, data):
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def append_log_row(row: dict):
    # ensure file exists with header
    if not LOG_FILE.exists():
        # create header if missing
        pd.DataFrame([row]).to_csv(LOG_FILE, index=False)
        return
    # append
    try:
        pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=False, index=False)
    except Exception as e:
        print("[ERROR] failed to append log:", e)
        # as fallback, try writing entire file (safe)
        if LOG_FILE.exists():
            try:
                prev = pd.read_csv(LOG_FILE)
            except Exception:
                prev = pd.DataFrame()
        else:
            prev = pd.DataFrame()
        new = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
        new.to_csv(LOG_FILE, index=False)

# ---------- デバッグ用：現在の state と簡易売買シミュレーション（終値ベース） ----------
def load_state():
    if not STATE_FILE.exists():
        return {"capital": INIT_CAPITAL, "positions": []}
    try:
        return json.load(open(STATE_FILE, "r", encoding="utf-8"))
    except Exception:
        # broken file -> reinit
        return {"capital": INIT_CAPITAL, "positions": []}

def simple_daily_step():
    """
    ※本関数は本来のモデルを代替する簡易シミュレーションです。
    目的：ファイル作成とログ保存の検証を最優先に行うための安全処理。
    実運用ロジック（モデル学習や yfinance 取得）をここに戻すことも可能です。
    """
    state = load_state()
    capital = state.get("capital", INIT_CAPITAL)
    positions = state.get("positions", [])
    run_date = datetime.datetime.now(TIMEZONE).date()
    market_date = run_date - datetime.timedelta(days=1)  # ダミー market_date

    # 土日チェック（JST）
    weekday = run_date.weekday()  # Mon=0
    if weekday >= 5:
        # 土日スキップだがログを書いておく
        row = {
            "run_date_jst": run_date.isoformat(),
            "market_date": market_date.isoformat(),
            "status": "weekend_skip",
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
        }
        append_log_row(row)
        print("[INFO] weekend - wrote skip log")
        return

    # --- ここから本来は yfinance 取得やモデルで売買判断を入れる ---
    # ここではテスト用に「何もしない」シナリオを実行し、状態を確実に保存してログに残す。
    # また例外時は CSV に error を残す。
    try:
        realized = 0.0
        unrealized = 0.0
        equity = capital + unrealized
        buys = []
        sells = []
        # Save state back (no change)
        state["capital"] = capital
        state["positions"] = positions
        safe_write_json(STATE_FILE, state)

        # compute daily return vs last log
        prev_equity = None
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
            "run_date_jst": run_date.isoformat(),
            "market_date": market_date.isoformat(),
            "status": "ok_no_trades",
            "realized_pnl": float(realized),
            "unrealized_pnl": float(unrealized),
            "equity": float(equity),
            "capital": float(capital),
            "num_positions": len(positions),
            "positions": json.dumps(positions, ensure_ascii=False),
            "buys": ";".join(buys),
            "sells": ";".join(sells),
            "daily_return_pct": float(daily_ret),
            "error": ""
        }
        append_log_row(row)
        print("[INFO] daily step completed - log appended")
    except Exception as e:
        # 捕捉した例外は CSV に書いて残す（デバッグ容易化）
        tb = traceback.format_exc()
        row = {
            "run_date_jst": run_date.isoformat(),
            "market_date": market_date.isoformat(),
            "status": "error",
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "equity": float(capital),
            "capital": float(capital),
            "num_positions": len(positions),
            "positions": json.dumps(positions, ensure_ascii=False),
            "buys": "",
            "sells": "",
            "daily_return_pct": 0.0,
            "error": str(e) + "\n" + tb[:4000]
        }
        append_log_row(row)
        print("[ERROR] exception occurred during daily step; logged to CSV")
        raise

# ---------- main ----------
def main():
    try:
        ensure_files_exist()
        simple_daily_step()
        print("[DONE] safe run finished")
    except Exception as e:
        print("[FATAL] unexpected error:", e)
        # 最低限、エラーログ行を残す
        run_date = datetime.datetime.now(TIMEZONE).date()
        try:
            row = {
                "run_date_jst": run_date.isoformat(),
                "market_date": "",
                "status": "fatal_error",
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "equity": float(INIT_CAPITAL),
                "capital": float(INIT_CAPITAL),
                "num_positions": 0,
                "positions": "",
                "buys": "",
                "sells": "",
                "daily_return_pct": 0.0,
                "error": str(e)
            }
            append_log_row(row)
        except Exception:
            print("[FATAL] couldn't write error log")

if __name__ == "__main__":
    main()















