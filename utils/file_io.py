import os, json, pandas as pd

def load_equity_state(path, init_capital):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"capital": init_capital, "positions": []}

def save_equity_state(path, state):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def append_csv_row(path, row_dict):
    df = pd.DataFrame([row_dict])
    if not os.path.exists(path):
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(path, index=False, mode="a", header=False, encoding="utf-8-sig")
