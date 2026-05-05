import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import PPO
from env.trading_env import TradingEnv
from features.technical import add_indicators
from features.sentiment import synthetic_sentiment
import yfinance as yf
from config import *

st.set_page_config(page_title="Sentiment Trading Agent", layout="wide", page_icon="🤖")

st.markdown("""
    <h1 style='text-align:center;'>Sentiment-Driven Algorithmic Trading Agent</h1>
    <p style='text-align:center; color:gray;'>Deep Reinforcement Learning + FinBERT NLP + Technical Analysis</p>
    <hr/>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[pd.to_numeric(df['Close'], errors='coerce').notna()].iloc[1:]
    df['Close']  = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df = add_indicators(df)
    df['sentiment'] = synthetic_sentiment(df).values
    df.dropna(inplace=True)
    split = int(len(df) * TRAIN_SPLIT)
    return df.iloc[split:].copy().reset_index(drop=True)

@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        m = PPO.load(os.path.join(base, "models", "best_model"))
        return m, "V2 Best Model (1M steps, 6yr data)"
    except:
        m = PPO.load(os.path.join(base, "models", "ppo_trader_v2"))
        return m, "V2 Final Model"

def load_trade_log():
    log_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "live", "trade_log.json"
    )
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return json.load(f)
    return {"trades": [], "portfolio_history": []}

with st.spinner("Loading data and model..."):
    test_df = load_data()
    model, model_name = load_model()

# Sidebar
st.sidebar.header("Configuration")
st.sidebar.markdown(f"**Ticker:** {TICKER}")
st.sidebar.markdown(f"**Initial Balance:** ${INITIAL_BALANCE:,}")
st.sidebar.markdown(f"**Window Size:** {WINDOW_SIZE} days")
st.sidebar.markdown(f"**Model:** {model_name}")
st.sidebar.markdown("---")
st.sidebar.markdown("### Signals Used")
st.sidebar.markdown("- Technical Indicators (RSI, MACD, BB)")
st.sidebar.markdown("- Sentiment (FinBERT NLP)")
st.sidebar.markdown("- PPO Policy Network")
st.sidebar.markdown("---")
st.sidebar.markdown("### V1 vs V2")
st.sidebar.markdown("| Metric | V1 | V2 |")
st.sidebar.markdown("|---|---|---|")
st.sidebar.markdown("| Return | 8.90% | **55.73%** |")
st.sidebar.markdown("| Sharpe | 2.618 | **4.300** |")
st.sidebar.markdown("| Drawdown | -3.41% | **-3.75%** |")

# Tabs
tab1, tab2 = st.tabs(["📊 Backtest", "🔴 Live Paper Trading"])

# ── TAB 1: BACKTEST ─────────────────────────────────
with tab1:
    run = st.button("Run Backtest", type="primary")

    if run:
        with st.spinner("Running backtest..."):
            env = TradingEnv(test_df, WINDOW_SIZE, INITIAL_BALANCE)
            obs, _ = env.reset()
            portfolio_values = [INITIAL_BALANCE]
            actions_log = []
            done = False
            while not done:
                obs_input = np.array(obs).reshape(1, -1)
                action, _ = model.predict(obs_input, deterministic=True)
                action = int(np.squeeze(action))
                obs, _, done, _, info = env.step(action)
                portfolio_values.append(info['net_worth'])
                actions_log.append(action)

        pv  = np.array(portfolio_values)
        ret = np.diff(pv) / pv[:-1]
        bh  = INITIAL_BALANCE * test_df['Close'].astype(float).values \
              / float(test_df['Close'].iloc[0])

        sharpe    = np.sqrt(252) * ret.mean() / (ret.std() + 1e-8)
        mdd       = ((pv - np.maximum.accumulate(pv)) \
                    / np.maximum.accumulate(pv)).min() * 100
        total_ret = (pv[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        bh_ret    = (float(test_df['Close'].iloc[-1]) \
                    - float(test_df['Close'].iloc[0])) \
                    / float(test_df['Close'].iloc[0]) * 100

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Final Portfolio", f"${pv[-1]:,.2f}", f"{total_ret:+.2f}%")
        col2.metric("Buy & Hold",      f"${bh[len(pv)-1]:,.2f}", f"{bh_ret:+.2f}%")
        col3.metric("Sharpe Ratio",    f"{sharpe:.3f}", "↑ vs 2.618 V1")
        col4.metric("Max Drawdown",    f"{mdd:.2f}%")
        col5.metric("Total Trades",    sum(1 for a in actions_log if a != 0))

        st.markdown("---")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.55, 0.25, 0.20],
                            subplot_titles=["Portfolio Value", "RSI", "Sentiment"])

        fig.add_trace(go.Scatter(y=pv, name="PPO Agent V2",
                                 line=dict(color="steelblue", width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(y=bh[:len(pv)], name="Buy & Hold",
                                 line=dict(color="orange", dash="dash", width=2)),
                      row=1, col=1)

        buy_steps  = [i for i, a in enumerate(actions_log) if a == 1]
        sell_steps = [i for i, a in enumerate(actions_log) if a == 2]
        fig.add_trace(go.Scatter(x=buy_steps, y=[pv[i] for i in buy_steps],
                                 mode='markers', name='Buy',
                                 marker=dict(symbol='triangle-up',
                                             color='green', size=10)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_steps, y=[pv[i] for i in sell_steps],
                                 mode='markers', name='Sell',
                                 marker=dict(symbol='triangle-down',
                                             color='red', size=10)),
                      row=1, col=1)

        rsi = test_df['rsi'].values[:len(actions_log)]
        fig.add_trace(go.Scatter(y=rsi, name="RSI",
                                 line=dict(color="purple", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red",   row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

        sentiment = test_df['sentiment'].values[:len(actions_log)]
        fig.add_trace(go.Bar(y=sentiment, name="Sentiment",
                             marker_color=['green' if s > 0 else 'red'
                                           for s in sentiment]),
                      row=3, col=1)

        fig.update_layout(height=750, template="plotly_dark",
                          title=f"PPO Agent V2 — {TICKER} | "
                                f"Sharpe: {sharpe:.3f} | Return: {total_ret:.2f}%")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Buy Actions",  actions_log.count(1))
        c2.metric("Hold Actions", actions_log.count(0))
        c3.metric("Sell Actions", actions_log.count(2))

        st.markdown("---")
        st.markdown("### How It Works")
        st.markdown("""
        | Component | Details |
        |---|---|
        | **Price Data** | AAPL OHLCV via yfinance (2018-2024) |
        | **Technical Indicators** | RSI, MACD, Bollinger Bands, Volume Ratio |
        | **Sentiment** | FinBERT NLP model (ProsusAI/finbert) |
        | **RL Algorithm** | PPO (Proximal Policy Optimization) |
        | **State Space** | 20-day window x 8 features = 160-dim vector |
        | **Action Space** | Discrete(3): Buy / Hold / Sell |
        | **Reward** | Log portfolio return with 0.1% commission |
        | **Training** | 1,000,000 steps over 6 years of data |
        """)
    else:
        st.info("Click **Run Backtest** to see the agent in action!")
        st.code("""
Financial News -> FinBERT NLP -> Sentiment Score [-1, +1]
Price Data     -> RSI, MACD, Bollinger Bands
               -> State Vector (160-dim)
               -> PPO Agent V2 -> Buy / Hold / Sell
               -> Backtest: Sharpe 4.300 | Return 55.73%
        """)

# ── TAB 2: LIVE PAPER TRADING ────────────────────────
with tab2:
    st.markdown("## 🔴 Live Paper Trading Monitor")
    st.markdown("Trades execute daily at **7:00 PM IST** (9:30 AM EST) via Alpaca Paper Trading")
    st.markdown("---")

    log = load_trade_log()

    if not log["portfolio_history"]:
        st.info("No live trades yet. The agent runs daily at market open (7 PM IST). Check back after the first trading session!")
        st.markdown("""
        ### How Paper Trading Works
        1. Every weekday at 7 PM IST, the scheduler runs automatically
        2. The agent fetches latest AAPL data
        3. It decides: Buy / Hold / Sell
        4. Order is placed on Alpaca Paper Trading (fake money, real market)
        5. Results appear here in real time
        """)
    else:
        # Portfolio history chart
        history = pd.DataFrame(log["portfolio_history"])
        history["timestamp"] = pd.to_datetime(history["timestamp"])

        # Key metrics
        initial = log["portfolio_history"][0]["portfolio_value"]
        latest  = log["portfolio_history"][-1]["portfolio_value"]
        ret     = (latest - initial) / initial * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Portfolio", f"${latest:,.2f}", f"{ret:+.2f}%")
        col2.metric("Starting Value",    f"${initial:,.2f}")
        col3.metric("Total Trades",      len(log["trades"]))
        col4.metric("Last Action",
                    log["portfolio_history"][-1]["action"])

        st.markdown("---")

        # Portfolio value over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history["timestamp"],
            y=history["portfolio_value"],
            name="Portfolio Value",
            line=dict(color="steelblue", width=2.5),
            fill='tozeroy',
            fillcolor='rgba(70,130,180,0.1)'
        ))

        # Mark trades
        if log["trades"]:
            trades_df = pd.DataFrame(log["trades"])
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
            buys  = trades_df[trades_df["action"] == "BUY"]
            sells = trades_df[trades_df["action"] == "SELL"]

            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys["timestamp"],
                    y=buys["portfolio_value"],
                    mode='markers', name='BUY',
                    marker=dict(symbol='triangle-up', color='green', size=14)
                ))
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells["timestamp"],
                    y=sells["portfolio_value"],
                    mode='markers', name='SELL',
                    marker=dict(symbol='triangle-down', color='red', size=14)
                ))

        fig.update_layout(
            title="Live Paper Portfolio — AAPL",
            template="plotly_dark",
            height=400,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Trade log table
        st.markdown("### 📋 Trade History")
        if log["trades"]:
            trades_df = pd.DataFrame(log["trades"])
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
            trades_df = trades_df.sort_values("timestamp", ascending=False)
            trades_df.columns = ["Time", "Action", "Shares", "Price", "Portfolio Value"]
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No buy/sell orders placed yet — agent has been holding.")

        # Daily log table
        st.markdown("### 📅 Daily Agent Decisions")
        display = history[["timestamp", "action", "price",
                           "portfolio_value", "cash", "shares"]].copy()
        display.columns = ["Time", "Decision", "AAPL Price",
                          "Portfolio Value", "Cash", "Shares"]
        display = display.sort_values("Time", ascending=False)
        st.dataframe(display, use_container_width=True)
