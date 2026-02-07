from fastapi import FastAPI, HTTPException
import joblib
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands
import os

app = FastAPI(title="QuantVision AI Engine")

# Load model using absolute path
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "stock_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError("Run train_model.py first to create stock_model.pkl")

model_data = joblib.load(model_path)
model = model_data["model"]
features_list = model_data["features"]

@app.get("/")
def home():
    return {"status": "online", "engine": "XGBoost-v2-High-Precision"}

@app.get("/predict/{ticker}")
def predict(ticker: str):
    try:
        # Fetch 1 year of data to calculate indicators accurately
        df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)
        if df.empty:
            raise HTTPException(status_code=404, detail="Ticker not found")
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        close_prices = df['Close'].squeeze()

        # Feature Engineering (Must match train_model.py exactly)
        df['SMA_50'] = SMAIndicator(close=close_prices, window=50).sma_indicator()
        df['SMA_200'] = SMAIndicator(close=close_prices, window=200).sma_indicator()
        df['RSI'] = RSIIndicator(close=close_prices, window=14).rsi()
        
        macd_init = MACD(close=close_prices)
        df['MACD'] = macd_init.macd()
        df['MACD_Signal'] = macd_init.macd_signal()

        bb_init = BollingerBands(close=close_prices, window=20, window_dev=2)
        df['BB_High'] = bb_init.bollinger_hband()
        df['BB_Low'] = bb_init.bollinger_lband()
        
        current_features = df[features_list].tail(1)
        
        if current_features.isnull().values.any():
            return {"error": "Technical indicators still calculating. Try a more liquid stock."}

        # Model Prediction
        prediction = int(model.predict(current_features)[0])
        prob = model.predict_proba(current_features)[0]

        return {
            "ticker": ticker.upper(),
            "prediction": "UP" if prediction == 1 else "DOWN",
            "confidence": round(float(max(prob)) * 100, 2),
            "current_price": round(float(close_prices.iloc[-1]), 2),
            "rsi": round(float(df['RSI'].iloc[-1]), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))