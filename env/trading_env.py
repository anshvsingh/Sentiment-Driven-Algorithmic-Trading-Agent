import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, window_size: int = 20,
                 initial_balance: float = 10_000, commission: float = 0.001):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission

        self.feature_cols = [
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'bb_pct', 'volume_ratio', 'returns', 'sentiment'
        ]
        n_features = len(self.feature_cols)

        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size * n_features,),
            dtype=np.float32
        )

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        mean = obs.mean(axis=0)
        std  = obs.std(axis=0) + 1e-8
        return ((obs - mean) / std).flatten()

    def _get_obs(self):
        window = self.df[self.feature_cols].iloc[
            self.current_step - self.window_size : self.current_step
        ].values
        return self._normalize(window).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step   = self.window_size
        self.balance        = self.initial_balance
        self.shares_held    = 0
        self.net_worth      = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.trades         = []
        return self._get_obs(), {}

    def step(self, action):
        price = float(self.df['Close'].iloc[self.current_step])

        if action == 1:  # BUY
            shares_to_buy = int(self.balance / (price * (1 + self.commission)))
            cost = shares_to_buy * price * (1 + self.commission)
            self.shares_held += shares_to_buy
            self.balance     -= cost
            if shares_to_buy > 0:
                self.trades.append({'step': self.current_step,
                                    'action': 'BUY', 'price': price})

        elif action == 2:  # SELL
            if self.shares_held > 0:
                revenue = self.shares_held * price * (1 - self.commission)
                self.balance     += revenue
                self.trades.append({'step': self.current_step,
                                    'action': 'SELL', 'price': price})
                self.shares_held  = 0

        self.net_worth      = self.balance + self.shares_held * price
        reward              = np.log(self.net_worth / self.prev_net_worth + 1e-8)
        self.prev_net_worth = self.net_worth
        self.current_step  += 1
        done = self.current_step >= len(self.df) - 1

        return self._get_obs(), float(reward), done, False, {
            'net_worth': self.net_worth,
            'balance':   self.balance,
            'shares':    self.shares_held
        }

    def render(self):
        profit = self.net_worth - self.initial_balance
        print(f"Step: {self.current_step} | Net Worth: ${self.net_worth:.2f} | "
              f"Profit: ${profit:.2f} | Shares: {self.shares_held}")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/anshvardhansingh/Desktop/trading_agent')
    import pandas as pd
    import numpy as np
    from features.technical import add_indicators
    from features.sentiment import synthetic_sentiment

    df = pd.read_csv("data/AAPL_ohlcv.csv", index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[pd.to_numeric(df['Close'], errors='coerce').notna()].iloc[1:]
    df['Close']  = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df = add_indicators(df)
    df['sentiment'] = synthetic_sentiment(df).values

    env = TradingEnv(df)
    obs, _ = env.reset()
    print(f"✅ Environment created")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Action space: {env.action_space}")

    # Random agent test
    total_reward = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

    print(f"   Random agent final net worth: ${info['net_worth']:,.2f}")
    print(f"   Total trades: {len(env.trades)}")
    print("✅ Environment test passed!")
