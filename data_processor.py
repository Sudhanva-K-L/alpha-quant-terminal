import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands

def build_stock_dataset(ticker):
    print(f"Fetching data for {ticker}...")
    
    # 1. Download Data
    df = yf.download(ticker, period="10y", interval="1d", auto_adjust=True)
    
    if df.empty:
        print(f"Error: No data found for {ticker}.")
        return pd.DataFrame()

    # Flatten columns if they are MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Use 'Close' price for calculations
    close_prices = df['Close'].squeeze()

    # 2. Add Basic Indicators (The ones you had)
    df['SMA_50'] = SMAIndicator(close=close_prices, window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(close=close_prices, window=200).sma_indicator()
    df['RSI'] = RSIIndicator(close=close_prices, window=14).rsi()

    # 3. Add Advanced Indicators (The ones causing the error)
    # MACD
    macd_init = MACD(close=close_prices)
    df['MACD'] = macd_init.macd()
    df['MACD_Signal'] = macd_init.macd_signal()

    # Bollinger Bands
    bb_init = BollingerBands(close=close_prices, window=20, window_dev=2)
    df['BB_High'] = bb_init.bollinger_hband()
    df['BB_Low'] = bb_init.bollinger_lband()
    
    # 4. Define the Target (Price in 5 days > current price)
    df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    
    # 5. Cleanup
    # Dropping NaNs is vital because indicators like SMA_200 need 200 days of data first
    df.dropna(inplace=True)
    
    print(f"Processing complete. Dataset shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Quick test to verify columns
    data = build_stock_dataset("AAPL")
    print("\nVerified Columns:", data.columns.tolist())