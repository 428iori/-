# utils_data.py
import yfinance as yf
import pandas as pd
from feature_engineer import make_features

def load_recent_data(tickers, start="2024-01-01"):
    all_data = []
    data = yf.download(tickers, start=start, progress=False, auto_adjust=True, group_by='ticker', threads=True)
    for t in tickers:
        if t not in data.columns.levels[0]: continue
        df = data[t].copy().reset_index()
        df = make_features(df)
        df["ticker"] = t
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)
