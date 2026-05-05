import yfinance as yf
import pandas as pd
from config import TICKER, START_DATE, END_DATE

def fetch_ohlcv(ticker=TICKER, start=START_DATE, end=END_DATE):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.dropna(inplace=True)
    df.to_csv(f"data/{ticker}_ohlcv.csv")
    print(f"✅ Downloaded {len(df)} rows for {ticker}")
    return df

if __name__ == "__main__":
    fetch_ohlcv()