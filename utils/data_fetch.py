import random

def get_execution_price_for_sell(ticker, date):
    """簡易売却価格: モックまたは実データ"""
    return random.uniform(0.98, 1.02) * 100  # 実際はyfinanceなどから取得

def get_execution_price_for_buy(ticker, date):
    return random.uniform(0.98, 1.02) * 100
