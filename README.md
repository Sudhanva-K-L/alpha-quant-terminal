# ⚖️ Alpha-Quant Predictive Terminal

An end-to-end financial intelligence system that uses Machine Learning to forecast short-term stock price movements.

## 🛠️ System Architecture
- **Predictive Engine:** XGBoost Classifier trained on 10 years of historical data.
- **Backend API:** FastAPI microservice for real-time inference.
- **Frontend:** Streamlit-based interactive dashboard with custom CSS.
- **Indicators:** MACD, RSI, Bollinger Bands, and Volume Flow.

## 🚀 Key Features
- **High-Precision Signals:** Optimized for an 84% precision rate on "Buy" signals.
- **Live News Ticker:** Real-time headline streaming via Yahoo Finance.
- **Volatility Analysis:** Interactive Bollinger Band overlays.

## 📈 How to Run
1. Start the API: `uvicorn app:app --port 8000`
2. Launch Terminal: `streamlit run dashboard.py`