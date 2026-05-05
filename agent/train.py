import sys
sys.path.insert(0, '/Users/anshvardhansingh/Desktop/trading_agent')

import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from env.trading_env import TradingEnv
from features.technical import add_indicators
from features.sentiment import synthetic_sentiment
from config import *

class ProgressCallback(BaseCallback):
    def __init__(self, check_freq=10000):
        super().__init__()
        self.check_freq = check_freq

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            print(f"  📈 Steps: {self.n_calls:,} / 200,000")
        return True

def build_dataset(ticker=TICKER):
    df = pd.read_csv(f"data/{ticker}_ohlcv.csv", index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[pd.to_numeric(df['Close'], errors='coerce').notna()].iloc[1:]
    df['Close']  = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df = add_indicators(df)
    df['sentiment'] = synthetic_sentiment(df).values
    df.dropna(inplace=True)
    print(f"✅ Dataset ready: {len(df)} rows, {len(df.columns)} features")
    return df

def train():
    df = build_dataset()

    split    = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()
    print(f"   Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    train_env = DummyVecEnv([lambda: TradingEnv(train_df, WINDOW_SIZE, INITIAL_BALANCE)])
    eval_env  = DummyVecEnv([lambda: TradingEnv(test_df,  WINDOW_SIZE, INITIAL_BALANCE)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=0
    )

    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log="logs/tensorboard/"
    )

    print("\n🚀 Training PPO agent (this takes ~10-20 mins)...")
    model.learn(
        total_timesteps=200_000,
        callback=[eval_callback, ProgressCallback()]
    )

    model.save(MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")
    return model, test_df

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)
    model, test_df = train()
    test_df.to_csv("data/test_df.csv")
    print("✅ Test data saved to data/test_df.csv")
