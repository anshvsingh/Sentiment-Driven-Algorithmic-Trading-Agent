import feedparser
import datetime
import pandas as pd
import numpy as np
from transformers import pipeline

print("⏳ Loading FinBERT model...")
finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    top_k=None
)
print("✅ FinBERT loaded")

def score_text(text: str) -> float:
    results = finbert(text[:512])[0]
    scores = {r['label']: r['score'] for r in results}
    return scores.get('positive', 0) - scores.get('negative', 0)

def synthetic_sentiment(df: pd.DataFrame) -> pd.Series:
    """
    Fallback: derive sentiment from price momentum + volatility.
    Positive momentum + low volatility = bullish sentiment.
    """
    close = pd.to_numeric(df['Close'], errors='coerce')
    momentum = close.pct_change(5).fillna(0)       # 5-day return
    volatility = close.pct_change().rolling(5).std().fillna(0)
    sentiment = momentum / (volatility + 1e-8)
    # Normalize to [-1, 1]
    sentiment = sentiment.clip(-3, 3) / 3
    print("✅ Using synthetic sentiment (price momentum based)")
    return pd.Series(sentiment.values, index=df.index, name='sentiment')

def fetch_news_sentiment(ticker: str, df: pd.DataFrame) -> pd.Series:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(url)

    records = []
    for entry in feed.entries:
        try:
            pub = datetime.datetime(*entry.published_parsed[:6])
            text = entry.title + " " + entry.get('summary', '')
            score = score_text(text)
            records.append({'date': pub.date(), 'sentiment': score})
        except Exception:
            continue

    if not records:
        return synthetic_sentiment(df)

    df_news = pd.DataFrame(records)
    df_news['date'] = pd.to_datetime(df_news['date'])
    daily = df_news.groupby('date')['sentiment'].mean()
    daily = daily.reindex(df.index).ffill().fillna(0)

    # Blend with synthetic for missing days
    synthetic = synthetic_sentiment(df)
    blended = daily.where(daily != 0, synthetic)
    print(f"✅ Sentiment ready: {len(records)} articles found, avg={blended.mean():.3f}")
    return blended

if __name__ == "__main__":
    df = pd.read_csv("data/AAPL_ohlcv.csv", index_col=0, parse_dates=True)
    df = df[pd.to_numeric(df['Close'], errors='coerce').notna()].iloc[1:]
    sentiment = fetch_news_sentiment("AAPL", df)
    print(sentiment.tail())
    print(f"Range: [{sentiment.min():.3f}, {sentiment.max():.3f}]")
