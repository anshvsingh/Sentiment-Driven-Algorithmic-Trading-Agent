import pandas as pd
import numpy as np

def synthetic_sentiment(df: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(df['Close'], errors='coerce')
    momentum = close.pct_change(5).fillna(0)
    volatility = close.pct_change().rolling(5).std().fillna(0)
    sentiment = momentum / (volatility + 1e-8)
    sentiment = sentiment.clip(-3, 3) / 3
    print("✅ Using synthetic sentiment")
    return pd.Series(sentiment.values, index=df.index, name='sentiment')

def fetch_news_sentiment(ticker: str, df: pd.DataFrame) -> pd.Series:
    return synthetic_sentiment(df)

def score_text(text: str) -> float:
    return 0.0
