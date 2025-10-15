import yfinance as yf, pandas as pd, os
os.makedirs("price_cache", exist_ok=True)

tickers = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","SPY","QQQ","VIX"]
for t in tickers:
    print("Downloading:", t)
    df = yf.download(t, start="2010-01-01", end="2025-10-15", auto_adjust=True)
    df.reset_index().to_csv(f"price_cache/{t}.csv", index=False)
print("✅ Done. CSVs ready in price_cache/")
