import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime

# Page Config
st.set_page_config(page_title="Alpha-Quant Terminal", page_icon="⚖️", layout="wide")

# 1. Institutional UI Styling (CSS Injection)
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    [data-testid="stMetricValue"] { font-family: 'Courier New', monospace; font-weight: bold; color: #00ff88 !important; }
    .stAlert { border-radius: 8px; border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    div.stButton > button { width: 100%; background-color: #238636; color: white; border-radius: 5px; height: 3em; font-weight: bold; border: none; }
    div.stButton > button:hover { background-color: #2ea043; border: none; }
    .metric-card { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# 2. Header Section
t1, t2 = st.columns([3, 1])
with t1:
    st.title("⚖️ Alpha-Quant Terminal")
    st.caption(f"Proprietary Engine v2.5 | High-Precision Mode | Server Time: {datetime.now().strftime('%H:%M:%S')}")
with t2:
    if st.button("🔄 REBOOT ENGINE"):
        st.rerun()

st.divider()

# 3. Sidebar Control Room
with st.sidebar:
    st.header("🔍 Market Ops")
    ticker = st.text_input("ASSET TICKER", value="AAPL").upper()
    
    st.markdown("---")
    st.subheader("Terminal View")
    chart_style = st.selectbox("Render Mode", ["Advanced Candlestick", "Baseline Overlay"])
    timeframe = st.select_slider("Data Depth", options=["30D", "90D", "180D", "1Y", "2Y"], value="180D")
    
    st.markdown("---")
    st.markdown("### 🛠️ Quick Toggles")
    show_volume = st.checkbox("Volume Flow", value=True)
    show_bands = st.checkbox("Volatility Bands", value=True)

# 4. Main Execution Logic
try:
    # Fetch Data from FastAPI
    api_url = f"https://web-production-ce9e6.up.railway.app/predict/{ticker}"
    res = requests.get(api_url).json()

    if "error" in res:
        st.error(f"SYSTEM OVERLOAD: {res['error']}")
    else:
        # A. Top Tier Metrics (Alpha Cards)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("LAST TRADED", f"${res['current_price']}")
        m2.metric("PREDICTIVE BIAS", res['prediction'])
        m3.metric("CONFIDENCE SCORE", f"{res['confidence']}%")
        m4.metric("RSI STRENGTH", res['rsi'])

        # B. Multi-Timeframe Performance (NEW FEATURE)
        st.markdown("### 📊 Momentum Heatmap")
        hist = yf.download(ticker, period="2y", interval="1d", auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
        
        # Calculate Returns
        ret_1d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100
        ret_1w = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-5]) - 1) * 100
        ret_1m = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-21]) - 1) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("1D RETURN", f"{ret_1d:.2f}%", delta=f"{ret_1d:.2f}%")
        c2.metric("1W RETURN", f"{ret_1w:.2f}%", delta=f"{ret_1w:.2f}%")
        c3.metric("1M RETURN", f"{ret_1m:.2f}%", delta=f"{ret_1m:.2f}%")

        # C. Signal Intelligence Feed
        st.markdown("### 📡 Intelligence Feed")
        if res['prediction'] == "UP" and res['confidence'] >= 80:
            st.success(f"💎 **GOLDEN ALPHA:** Institutional-grade Buy Pattern detected. Confidence: {res['confidence']}%")
        elif res['prediction'] == "DOWN" and res['confidence'] >= 80:
            st.error(f"🔴 **EXIT COMMAND:** Strong Bearish Reversal detected. Reduce Exposure.")
        else:
            st.info(f"📊 **NEUTRAL FLOW:** Model indicates {res['prediction']} movement. High noise detected.")

        # D. Advanced Dual-Pane Charting (NEW FEATURE)
        st.markdown("---")
        tf_map = {"30D": 30, "90D": 90, "180D": 180, "1Y": 365, "2Y": 730}
        plot_df = hist.tail(tf_map[timeframe])

        # Create subplots for Price + Volume
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

        # Price Plot
        if chart_style == "Advanced Candlestick":
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                                        low=plot_df['Low'], close=plot_df['Close'], name='Price'), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], line=dict(color='#58a6ff', width=2), name='Close'), row=1, col=1)

        # Volatility Bands Overlay
        if show_bands:
            sma = plot_df['Close'].rolling(window=20).mean()
            std = plot_df['Close'].rolling(window=20).std()
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            fig.add_trace(go.Scatter(x=plot_df.index, y=upper, line=dict(color='rgba(173, 216, 230, 0.2)'), name='Upper Band'), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=lower, line=dict(color='rgba(173, 216, 230, 0.2)'), fill='tonexty', name='Lower Band'), row=1, col=1)

        # Volume Plot
        if show_volume:
            colors = ['#ff4b4b' if row['Open'] > row['Close'] else '#00ff88' for _, row in plot_df.iterrows()]
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=700, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning("⚠️ **ENGINE STANDBY:** Ensure the FastAPI engine is live on Port 8000.")
    st.caption(f"Log: {e}")
