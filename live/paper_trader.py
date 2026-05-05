import sys
sys.path.insert(0, '/Users/anshvardhansingh/Desktop/trading_agent')

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from stable_baselines3 import PPO
from features.technical import add_indicators
from features.sentiment import synthetic_sentiment
from env.trading_env import TradingEnv
from config import *

API_KEY    = os.getenv("ALPACA_API_KEY", "PK7UKSFFY2CHFHZH57VXUIAJYQ")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "6WeB4vYmnortNngJJU3H1gfdhiQAN47CzantSx1FcUQk")
LOG_FILE   = "live/trade_log.json"

client = TradingClient(API_KEY, SECRET_KEY, paper=True)
model  = PPO.load("models/best_model")
print("✅ Model loaded")

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return {"trades": [], "portfolio_history": []}

def save_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

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
        pos = client.get_open_position(ticker)
        return int(pos.qty)
    except:
        return 0

def get_cash():
    account = client.get_account()
    return float(account.cash)

def get_portfolio_value():
    account = client.get_account()
    return float(account.portfolio_value)

def get_agent_action(df):
    env = TradingEnv(df, WINDOW_SIZE, INITIAL_BALANCE)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
    action, _ = model.predict(obs, deterministic=True)
    return int(np.squeeze(action))

def is_market_open():
    clock = client.get_clock()
    return clock.is_open

def run():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*50}")
    print(f"🤖 Trading Agent Run — {now}")
    print(f"{'='*50}")

    if not is_market_open():
        print("⏰ Market is closed — skipping")
        return

    df = get_latest_data()
    if len(df) < WINDOW_SIZE + 1:
        print("❌ Not enough data")
        return

    action        = get_agent_action(df)
    price         = float(df['Close'].iloc[-1])
    cash          = get_cash()
    shares_held   = get_position()
    portfolio_val = get_portfolio_value()
    action_name   = ['HOLD', 'BUY', 'SELL'][action]

    print(f"  Price:       ${price:.2f}")
    print(f"  Cash:        ${cash:,.2f}")
    print(f"  Shares:      {shares_held}")
    print(f"  Portfolio:   ${portfolio_val:,.2f}")
    print(f"  Decision:    {action_name}")

    log = load_log()
    log["portfolio_history"].append({
        "timestamp":       now,
        "portfolio_value": portfolio_val,
        "cash":            cash,
        "shares":          shares_held,
        "price":           price,
        "action":          action_name
    })

    order_placed = False

    if action == 1 and shares_held == 0:  # BUY
        qty = int(cash * 0.95 / price)
        if qty > 0:
            order = MarketOrderRequest(
                symbol=TICKER,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            client.submit_order(order)
            print(f"  ✅ BUY {qty} shares @ ${price:.2f}")
            log["trades"].append({
                "timestamp":       now,
                "action":          "BUY",
                "qty":             qty,
                "price":           price,
                "portfolio_value": portfolio_val
            })
            order_placed = True

    elif action == 2 and shares_held > 0:  # SELL
        order = MarketOrderRequest(
            symbol=TICKER,
            qty=shares_held,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        client.submit_order(order)
        print(f"  ✅ SELL {shares_held} shares @ ${price:.2f}")
        log["trades"].append({
            "timestamp":       now,
            "action":          "SELL",
            "qty":             shares_held,
            "price":           price,
            "portfolio_value": portfolio_val
        })
        order_placed = True

    if not order_placed:
        print(f"  ⏸  HOLD — no order placed")

    save_log(log)
    print(f"  📝 Log saved")
    print(f"{'='*50}")

if __name__ == "__main__":
    run()
