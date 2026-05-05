import sys
sys.path.insert(0, '/Users/anshvardhansingh/Desktop/trading_agent')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.trading_env import TradingEnv
from config import *

def sharpe_ratio(returns, risk_free=0.0):
    excess = returns - risk_free / 252
    return np.sqrt(252) * excess.mean() / (excess.std() + 1e-8)

def max_drawdown(portfolio_values):
    peak = np.maximum.accumulate(portfolio_values)
    dd   = (portfolio_values - peak) / peak
    return dd.min()

def run_backtest():
    test_df = pd.read_csv("data/test_df.csv", index_col=0, parse_dates=True)

    # Try best model first, fallback to last
    try:
        model = PPO.load("models/best_model")
        print("✅ Loaded best model")
    except:
        model = PPO.load(MODEL_PATH)
        print("✅ Loaded final model")

    env = TradingEnv(test_df, WINDOW_SIZE, INITIAL_BALANCE)
    obs, _ = env.reset()

    portfolio_values = [INITIAL_BALANCE]
    actions_log      = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        portfolio_values.append(info['net_worth'])
        actions_log.append(int(action))

    pv  = np.array(portfolio_values)
    ret = np.diff(pv) / pv[:-1]

    # Buy-and-hold benchmark
    bh_start  = float(test_df['Close'].iloc[0])
    bh_end    = float(test_df['Close'].iloc[-1])
    bh_return = (bh_end - bh_start) / bh_start * 100
    bh_values = INITIAL_BALANCE * test_df['Close'].astype(float).values / bh_start

    agent_return = (pv[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    print("\n" + "="*45)
    print("       📊 BACKTEST RESULTS")
    print("="*45)
    print(f"  Initial Balance   : ${INITIAL_BALANCE:,.2f}")
    print(f"  Final Portfolio   : ${pv[-1]:,.2f}")
    print(f"  Agent Return      : {agent_return:.2f}%")
    print(f"  Buy & Hold Return : {bh_return:.2f}%")
    print(f"  Sharpe Ratio      : {sharpe_ratio(ret):.3f}")
    print(f"  Max Drawdown      : {max_drawdown(pv)*100:.2f}%")
    print(f"  Total Trades      : {sum(1 for a in actions_log if a != 0)}")
    print(f"  Action Breakdown  : Buy={actions_log.count(1)} | "
          f"Hold={actions_log.count(0)} | Sell={actions_log.count(2)}")
    print("="*45)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(pv, label='🤖 PPO Agent', color='steelblue', linewidth=2)
    ax1.plot(bh_values[:len(pv)], label='📈 Buy & Hold',
             color='orange', linewidth=2, linestyle='--')
    ax1.set_title(f'PPO Agent vs Buy-and-Hold — {TICKER}', fontsize=14)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(); ax1.grid(alpha=0.3)

    # Action markers
    buy_steps  = [i for i, a in enumerate(actions_log) if a == 1]
    sell_steps = [i for i, a in enumerate(actions_log) if a == 2]
    ax1.scatter(buy_steps,  [pv[i] for i in buy_steps],
                marker='^', color='green', s=80, zorder=5, label='Buy')
    ax1.scatter(sell_steps, [pv[i] for i in sell_steps],
                marker='v', color='red',   s=80, zorder=5, label='Sell')

    # Sentiment subplot
    sentiment = test_df['sentiment'].values[:len(actions_log)]
    colors = ['green' if s > 0 else 'red' for s in sentiment]
    ax2.bar(range(len(sentiment)), sentiment, color=colors, alpha=0.6)
    ax2.set_ylabel('Sentiment'); ax2.set_xlabel('Trading Step')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("backtest/results.png", dpi=150)
    print("\n✅ Chart saved to backtest/results.png")
    plt.show()

if __name__ == "__main__":
    run_backtest()
