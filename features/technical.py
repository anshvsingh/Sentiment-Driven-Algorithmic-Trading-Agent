import pandas as pd
import numpy as np
import ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df['Close'], window=20)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_pct'] = bb.bollinger_pband()

    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']

    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/AAPL_ohlcv.csv", index_col=0, parse_dates=True)
    # Fix yfinance multi-level header
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[pd.to_numeric(df['Close'], errors='coerce').notna()]
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df = add_indicators(df)
    print(df[['Close', 'rsi', 'macd', 'bb_pct', 'volume_ratio']].tail())
    print(f"✅ Indicators added, {len(df)} rows remaining")