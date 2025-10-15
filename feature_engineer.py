# feature_engineer.py
import pandas as pd
import numpy as np

def rsi(series, n=14):
    diff = series.diff()
    up, down = diff.clip(lower=0), -diff.clip(upper=0)
    ma_up, ma_down = up.ewm(alpha=1/n, adjust=False).mean(), down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def make_features(df):
    c, v = df["Close"], df["Volume"]
    df["sma_s"] = c.rolling(10).mean()
    df["sma_m"] = c.rolling(30).mean()
    df["sma_l"] = c.rolling(90).mean()
    df["rsi"] = rsi(c, 14)
    df["ret1"] = c.pct_change()
    df["ret5"] = c.pct_change(5)
    df["vol_chg"] = v.pct_change()
    df["ma_gap"] = (c - df["sma_m"]) / (df["sma_m"] + 1e-12)
    df["volatility"] = c.pct_change().rolling(20).std()
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    df["macd"], df["macd_signal"] = macd, macd - macd.ewm(span=9).mean()
    df["momentum20"] = c / c.shift(20) - 1
    df = df.dropna().reset_index(drop=True)
    return df
