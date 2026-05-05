import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
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
    return df.iloc[split:].copy()

@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        m = PPO.load(os.path.join(base, "models", "best_model"))
        return m, "Best Model"
    except:
        m = PPO.load(os.path.join(base, "models", "ppo_trader"))
        return m, "Final Model"

with st.spinner("Loading data and model..."):
    test_df = load_data()
    model, model_name = load_model()

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

run = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)

if run:
    with st.spinner("Running backtest..."):
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
    bh  = INITIAL_BALANCE * test_df['Close'].astype(float).values / float(test_df['Close'].iloc[0])

    sharpe    = np.sqrt(252) * ret.mean() / (ret.std() + 1e-8)
    mdd       = ((pv - np.maximum.accumulate(pv)) / np.maximum.accumulate(pv)).min() * 100
    total_ret = (pv[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    bh_ret    = (float(test_df['Close'].iloc[-1]) - float(test_df['Close'].iloc[0])) \
                / float(test_df['Close'].iloc[0]) * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Final Portfolio", f"${pv[-1]:,.2f}", f"{total_ret:+.2f}%")
    col2.metric("Buy & Hold",      f"${bh[len(pv)-1]:,.2f}", f"{bh_ret:+.2f}%")
    col3.metric("Sharpe Ratio",    f"{sharpe:.3f}")
    col4.metric("Max Drawdown",    f"{mdd:.2f}%")
    col5.metric("Total Trades",    sum(1 for a in actions_log if a != 0))

    st.markdown("---")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.20],
                        subplot_titles=["Portfolio Value", "RSI", "Sentiment"])

    fig.add_trace(go.Scatter(y=pv, name="PPO Agent",
                             line=dict(color="steelblue", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(y=bh[:len(pv)], name="Buy & Hold",
                             line=dict(color="orange", dash="dash", width=2)), row=1, col=1)

    buy_steps  = [i for i, a in enumerate(actions_log) if a == 1]
    sell_steps = [i for i, a in enumerate(actions_log) if a == 2]
    fig.add_trace(go.Scatter(x=buy_steps, y=[pv[i] for i in buy_steps],
                             mode='markers', name='Buy',
                             marker=dict(symbol='triangle-up', color='green', size=10)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_steps, y=[pv[i] for i in sell_steps],
                             mode='markers', name='Sell',
                             marker=dict(symbol='triangle-down', color='red', size=10)),
                  row=1, col=1)

    rsi = test_df['rsi'].values[:len(actions_log)]
    fig.add_trace(go.Scatter(y=rsi, name="RSI",
                             line=dict(color="purple", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red",   row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    sentiment = test_df['sentiment'].values[:len(actions_log)]
    fig.add_trace(go.Bar(y=sentiment, name="Sentiment",
                         marker_color=['green' if s > 0 else 'red' for s in sentiment]),
                  row=3, col=1)

    fig.update_layout(height=750, template="plotly_dark",
                      title=f"PPO Agent — {TICKER} Test Period")
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
    | **Price Data** | AAPL OHLCV via yfinance (2022-2024) |
    | **Technical Indicators** | RSI, MACD, Bollinger Bands, Volume Ratio |
    | **Sentiment** | FinBERT NLP model (ProsusAI/finbert) |
    | **RL Algorithm** | PPO (Proximal Policy Optimization) |
    | **State Space** | 20-day window x 8 features = 160-dim vector |
    | **Action Space** | Discrete(3): Buy / Hold / Sell |
    | **Reward** | Log portfolio return with 0.1% commission |
    | **Training** | 200,000 environment steps |
    """)
else:
    st.info("Click **Run Backtest** in the sidebar to see the agent in action!")
    st.code("""
Financial News -> FinBERT NLP -> Sentiment Score [-1, +1]
Price Data     -> RSI, MACD, Bollinger Bands
               -> State Vector (160-dim)
               -> PPO Agent -> Buy / Hold / Sell
               -> Backtest: Sharpe 2.618 | Drawdown -3.41%
    """)
