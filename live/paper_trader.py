import sys
sys.path.insert(0, '/Users/anshvardhansingh/Desktop/trading_agent')

# --- New Alpaca Imports ---
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
# --------------------------

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from features.technical import add_indicators
from features.sentiment import synthetic_sentiment
from env.trading_env import TradingEnv
from config import *

# ── Alpaca credentials ──────────────────────────────
API_KEY    = "PK7UKSFFY2CHFHZH57VXUIAJYQ"
SECRET_KEY = "6WeB4vYmnortNngJJU3H1gfdhiQAN47CzantSx1FcUQk"
# In alpaca-py, the URL is handled by the 'paper' boolean in the client
# ────────────────────────────────────────────────────

# Initialize New Trading Client
api = TradingClient(API_KEY, SECRET_KEY, paper=True)

model = PPO.load("models/best_model")
print("✅ Model loaded")

def get_latest_data(ticker=TICKER, days=80):
    df = yf.download(ticker, period=f"{days}d", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[pd.to_numeric(df['Close'], errors='coerce').notna()].iloc[1:]
    df['Close']  = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df = add_indicators(df)
    df['sentiment'] = synthetic_sentiment(df).values
    df.dropna(inplace=True)
    return df

def get_position(ticker=TICKER):
    try:
        # In alpaca-py, it's get_open_position
        pos = api.get_open_position(ticker)
        return int(pos.qty)
    except:
        return 0

def get_cash():
    # Account attributes are now accessed via account object
    return float(api.get_account().cash)

def get_agent_action(df):
    """Run agent through history, return final action"""
    env = TradingEnv(df, WINDOW_SIZE, INITIAL_BALANCE)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
    action, _ = model.predict(obs, deterministic=True)
    return int(action)

def run():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*50}")
    print(f"🤖 Trading Agent Run — {now}")
    print(f"{'='*50}")

    # Check market is open
    clock = api.get_clock()
    if not clock.is_open:
        print("⏰ Market is closed — skipping")
        return

    # Get data and decide
    df = get_latest_data()
    if len(df) < WINDOW_SIZE + 1:
        print("❌ Not enough data")
        return

    action      = get_agent_action(df)
    price       = float(df['Close'].iloc[-1])
    cash        = get_cash()
    shares_held = get_position()
    portfolio   = cash + shares_held * price

    action_name = ['HOLD', 'BUY', 'SELL'][action]
    print(f"  Price:       ${price:.2f}")
    print(f"  Cash:        ${cash:,.2f}")
    print(f"  Shares:      {shares_held}")
    print(f"  Portfolio:   ${portfolio:,.2f}")
    print(f"  Decision:    {action_name}")

    if action == 1 and shares_held == 0:  # BUY
        qty = int(cash * 0.95 / price)
        if qty > 0:
            # New Order structure using MarketOrderRequest
            buy_order_data = MarketOrderRequest(
                symbol=TICKER,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            api.submit_order(order_data=buy_order_data)
            print(f"  ✅ BUY {qty} shares of {TICKER} @ ${price:.2f}")
        else:
            print("  ⚠️  Not enough cash to buy")

    elif action == 2 and shares_held > 0:  # SELL
        # New Order structure using MarketOrderRequest
        sell_order_data = MarketOrderRequest(
            symbol=TICKER,
            qty=shares_held,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        api.submit_order(order_data=sell_order_data)
        print(f"  ✅ SELL {shares_held} shares of {TICKER} @ ${price:.2f}")

    else:
        print(f"  ⏸  No order placed")

    print(f"{'='*50}")

if __name__ == "__main__":
    run()