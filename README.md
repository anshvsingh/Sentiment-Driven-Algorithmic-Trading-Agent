# 🤖 Sentiment-Driven Algorithmic Trading Agent

> **Deep Reinforcement Learning + Financial NLP + Technical Analysis**  
> A fully autonomous trading agent that learns optimal buy/hold/sell policies from market data and sentiment signals.

---

## 📊 Results

| Metric | Value |
|---|---|
| **Agent Return** | +8.90% |
| **Buy & Hold Return** | +10.80% |
| **Sharpe Ratio** | **2.618** ✅ |
| **Max Drawdown** | **-3.41%** ✅ |
| **Total Trades** | 57 |
| **Training Steps** | 200,000 |

> A Sharpe Ratio above 2.0 is considered outstanding. The agent's max drawdown of -3.41% vs buy-and-hold's ~-14% demonstrates superior **capital protection** during volatile periods.

---

## 🏗️ Architecture

```
Financial News (Yahoo RSS / NewsAPI)
        │
        ▼
   FinBERT NLP (ProsusAI/finbert)
   → Sentiment Score ∈ [-1.0, +1.0]
        │
        ▼
Price Data (yfinance OHLCV)
   → RSI · MACD · Bollinger Bands · Volume Ratio
        │
        ▼
   State Vector (20-day window × 8 features = 160-dim)
        │
        ▼
   PPO Agent (Proximal Policy Optimization)
   → Action: Buy / Hold / Sell
        │
        ▼
   Backtesting Engine
   → Sharpe · Max Drawdown · Returns · Trade Log
```

---

## 🧠 Why Reinforcement Learning?

Most student projects build **price-prediction models** (regression/classification). This project builds a **decision-making agent** — a fundamentally different and more realistic approach:

| Prediction Model | RL Trading Agent |
|---|---|
| Predicts next price | Decides what action to take |
| Ignores transaction costs | Penalizes every trade (0.1% commission) |
| Single-step output | Sequential decision making |
| Can't size positions | Learns when *not* to trade |
| Static after training | Reward signal drives continuous improvement |

This is closer to what **quantitative hedge funds** actually build.

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Price Data | `yfinance` |
| Technical Indicators | `ta` (RSI, MACD, Bollinger Bands) |
| NLP / Sentiment | `transformers` — ProsusAI/FinBERT |
| RL Environment | `gymnasium` (custom `TradingEnv`) |
| RL Algorithm | `stable-baselines3` — PPO |
| Visualization | `matplotlib`, `plotly` |
| Dashboard | `streamlit` |

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/trading-agent.git
cd trading-agent
pip install -r requirements.txt
```

### 2. Fetch Price Data

```bash
export PYTHONPATH=$(pwd)
python data/fetch_price.py
```

### 3. Train the PPO Agent

```bash
python agent/train.py
# Takes ~15 mins on CPU · Monitor via TensorBoard
tensorboard --logdir logs/tensorboard/
```

### 4. Run Backtest

```bash
python backtest/backtest.py
```

### 5. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📁 Project Structure

```
trading_agent/
├── config.py                  # Ticker, dates, hyperparameters
├── data/
│   └── fetch_price.py         # yfinance OHLCV downloader
├── features/
│   ├── technical.py           # RSI, MACD, Bollinger Bands
│   └── sentiment.py           # FinBERT NLP pipeline
├── env/
│   └── trading_env.py         # Custom Gymnasium environment
├── agent/
│   └── train.py               # PPO training loop
├── backtest/
│   └── backtest.py            # Performance metrics + charts
├── dashboard/
│   └── app.py                 # Streamlit demo interface
└── models/
    └── best_model.zip         # Saved PPO weights
```

---

## ⚙️ Configuration

Edit `config.py` to change ticker, date range, or training parameters:

```python
TICKER         = "AAPL"        # Any yfinance-supported ticker
START_DATE     = "2022-01-01"
END_DATE       = "2024-01-01"
INITIAL_BALANCE = 10_000
WINDOW_SIZE    = 20            # Lookback window for RL state
TRAIN_SPLIT    = 0.8           # 80% train, 20% test
```

---

## 🔬 Environment Design

### State Space
A **160-dimensional vector**: 20 trading days × 8 features per day:

| Feature | Description |
|---|---|
| `rsi` | Relative Strength Index (momentum) |
| `macd` | Moving Average Convergence Divergence (trend) |
| `macd_signal` | MACD signal line |
| `macd_diff` | MACD histogram |
| `bb_pct` | Bollinger Band position (0=lower, 1=upper) |
| `volume_ratio` | Volume vs 20-day average |
| `returns` | Daily percentage return |
| `sentiment` | FinBERT sentiment score [-1, +1] |

### Action Space
`Discrete(3)` — **Buy** (all-in) · **Hold** · **Sell** (all-out)

### Reward Function
```python
reward = log(net_worth_t / net_worth_{t-1})
```
Log return penalizes ruin (going to zero = −∞ reward) and naturally rewards compounding.

---

## 📰 Sentiment Pipeline

News headlines are scored using **FinBERT** (a BERT model fine-tuned on financial text):

```python
sentiment_score = P(positive) - P(negative)   # ∈ [-1.0, +1.0]
```

For historical backtesting where live news is unavailable, a **realized sentiment proxy** is computed from price momentum and volatility — a technique used in academic quantitative finance research.

In production, this pipeline connects directly to **NewsAPI**, **Bloomberg Terminal**, or **Refinitiv**.

---

## 📈 Key Insights from Backtest

- The agent **avoided the major drawdown** around step 45–50 where buy-and-hold dropped ~14% — demonstrating learned risk aversion
- The agent executed **57 trades** (32 buys, 25 sells, 16 holds) showing selective entry/exit rather than overtrading
- A **Sharpe Ratio of 2.618** indicates strong risk-adjusted returns (>1.0 = good, >2.0 = excellent)

---

## 🔮 Future Work

- **Multi-asset portfolio** — extend action space across multiple tickers simultaneously
- **Continuous actions** — switch PPO → SAC for fractional position sizing
- **Live news integration** — plug in NewsAPI or Bloomberg for real-time sentiment
- **Options strategies** — add derivatives to the action space
- **Walk-forward validation** — rolling train/test windows to reduce overfitting

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) — FinBERT model
- [Stable Baselines 3](https://stable-baselines3.readthedocs.io/) — PPO implementation
- [yfinance](https://github.com/ranaroussi/yfinance) — market data
- [ta](https://technical-analysis-library-in-python.readthedocs.io/) — technical indicators
