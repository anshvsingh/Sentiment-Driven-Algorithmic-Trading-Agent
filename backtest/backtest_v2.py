import sys
sys.path.insert(0, '/Users/anshvardhansingh/Desktop/trading_agent')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.trading_env import TradingEnv
from config import *

def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)

def max_drawdown(portfolio_values):
    peak = np.maximum.accumulate(portfolio_values)
    return ((portfolio_values - peak) / peak).min()

def run_backtest(model_path, test_csv):
    test_df = pd.read_csv(test_csv, index_col=0, parse_dates=True)

    try:
        model = PPO.load("models/best_model")
        print("✅ Loaded best model")
    except:
        model = PPO.load(model_path)
        print("✅ Loaded final model")

    env = TradingEnv(test_df, WINDOW_SIZE, INITIAL_BALANCE)
    obs, _ = env.reset()

    portfolio_values = [INITIAL_BALANCE]
    actions_log = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        portfolio_values.append(info['net_worth'])
        actions_log.append(int(action))

    pv  = np.array(portfolio_values)
    ret = np.diff(pv) / pv[:-1]

    bh_start = float(test_df['Close'].iloc[0])
    bh_end   = float(test_df['Close'].iloc[-1])
    bh_ret   = (bh_end - bh_start) / bh_start * 100
    bh_vals  = INITIAL_BALANCE * test_df['Close'].astype(float).values / bh_start

    agent_ret = (pv[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    print("\n" + "="*45)
    print("       📊 V2 BACKTEST RESULTS")
    print("="*45)
    print(f"  Initial Balance   : ${INITIAL_BALANCE:,.2f}")
    print(f"  Final Portfolio   : ${pv[-1]:,.2f}")
    print(f"  Agent Return      : {agent_ret:.2f}%")
    print(f"  Buy & Hold Return : {bh_ret:.2f}%")
    print(f"  Sharpe Ratio      : {sharpe_ratio(ret):.3f}")
    print(f"  Max Drawdown      : {max_drawdown(pv)*100:.2f}%")
    print(f"  Total Trades      : {sum(1 for a in actions_log if a != 0)}")
    print("="*45)

    # Compare v1 vs v2
    print("\n📊 V1 vs V2 Comparison")
    print("="*45)
    print(f"  {'Metric':<20} {'V1':>8} {'V2':>8}")
    print(f"  {'Agent Return':<20} {'8.90%':>8} {agent_ret:>7.2f}%")
    print(f"  {'Sharpe Ratio':<20} {'2.618':>8} {sharpe_ratio(ret):>8.3f}")
    print(f"  {'Max Drawdown':<20} {'-3.41%':>8} {max_drawdown(pv)*100:>7.2f}%")
    print("="*45)

    # Plot
    plt.figure(figsize=(14, 5))
    plt.plot(pv, label='PPO Agent V2', color='steelblue', linewidth=2)
    plt.plot(bh_vals[:len(pv)], label='Buy & Hold',
             color='orange', linewidth=2, linestyle='--')
    plt.title(f"PPO Agent V2 vs Buy-and-Hold — {TICKER}")
    plt.xlabel("Trading Steps")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("backtest/results_v2.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    run_backtest(MODEL_PATH, "data/test_df_v2.csv")
