#!/usr/bin/env python3
"""
Ultimate Stock Market AI Agent - Interactive Dashboard

Comprehensive Streamlit app that integrates with all our AI models and prediction systems.
Features live predictions, backtesting, portfolio analysis, and market insights.

Author: AI Assistant  
Date: July 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
import sys
import os
import json
import time

# Add src directory to path
sys.path.append('src')

try:
    from live_market_predictor import LiveMarketPredictor
    from comprehensive_predictor import ComprehensivePredictor
    from backtesting_framework import BacktestingEngine, MomentumStrategy, MeanReversionStrategy
    from app_utils import (
        LoadingState, handle_error, show_progress, format_large_number,
        create_info_card, validate_ticker, create_metric_card, 
        check_market_hours, create_disclaimer, format_timeframe
    )
except ImportError as e:
    st.error(f"❌ Could not import modules: {e}")
    st.info("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    st.stop()

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
page_title="StockSensei",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .live-badge {
        background: linear-gradient(90deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        animation: pulse 2s infinite;
        display: inline-block;
        margin: 0.5rem;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .bullish-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .bearish-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #333;
    }
    .metric-box {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        padding: 2rem 1.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        margin: 0.75rem 0;
        text-align: center;
        border: 2px solid #dee2e6;
        transition: all 0.3s ease;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        position: relative;
        overflow: hidden;
    }
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.18);
        border-color: #ced4da;
    }
    .metric-box::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102,126,234,0.05) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    .metric-box h3 {
        color: #212529 !important;
        font-weight: 900;
        font-size: 2.25rem;
        margin: 0 0 0.75rem 0;
        text-align: center;
        letter-spacing: -1px;
        line-height: 1;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        z-index: 1;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    .metric-box p {
        color: #495057 !important;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-align: center;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        z-index: 1;
    }
    .metric-box .metric-value {
        color: #111827;
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        display: block;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #667eea;
        margin: 1rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions_cache' not in st.session_state:
    st.session_state.predictions_cache = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'loading_state' not in st.session_state:
    st.session_state.loading_state = {}
if 'error_messages' not in st.session_state:
    st.session_state.error_messages = []

@st.cache_resource
@handle_error
def load_predictor():
    """Load the live market predictor"""
    with LoadingState("Loading AI prediction model..."):
        predictor = LiveMarketPredictor(
            model_path='models/random_forest_model.pkl',
            feature_names_path='models/feature_names.pkl'
        )
        return predictor

@st.cache_resource
def load_comprehensive_predictor():
    """Load the comprehensive predictor"""
    try:
        return ComprehensivePredictor()
    except Exception as e:
        st.error(f"❌ Error loading comprehensive predictor: {e}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_tickers():
    """Get list of available tickers from training data"""
    try:
        data = pd.read_csv('data/processed_features.csv')
        return sorted(data['Ticker'].unique())
    except:
        return ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']

def create_prediction_visualization(predictions_data):
    """Create comprehensive prediction visualizations"""
    if not predictions_data:
        return None, None
    
    # Extract data for visualization
    tickers = []
    current_prices = []
    predicted_prices = []
    changes = []
    confidences = []
    
    for ticker, result in predictions_data.items():
        if 'error' not in result:
            tickers.append(ticker)
            current_prices.append(result['current_price'])
            predicted_prices.append(result['predicted_price'])
            changes.append(result['price_change_pct'])
            confidences.append(result['confidence'])
    
    if not tickers:
        return None, None
    
    # Create subplots
    fig1 = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Price Comparison', 'Expected Changes (%)', 'Confidence Levels', 'Price Distribution'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Price comparison chart
    fig1.add_trace(
        go.Bar(x=tickers, y=current_prices, name='Current Price', 
               marker_color='lightblue', text=[f'${p:.2f}' for p in current_prices],
               textposition='auto', textfont=dict(color='white', size=12, family='Arial')),
        row=1, col=1
    )
    fig1.add_trace(
        go.Bar(x=tickers, y=predicted_prices, name='Predicted Price',
               marker_color=['red' if c < 0 else 'green' for c in changes],
               text=[f'${p:.2f}' for p in predicted_prices],
               textposition='auto', textfont=dict(color='white', size=12, family='Arial')),
        row=1, col=1
    )
    
    # Expected changes chart
    fig1.add_trace(
        go.Bar(x=tickers, y=changes, name='Expected Change %',
               marker_color=['red' if c < 0 else 'green' for c in changes],
               text=[f'{c:+.1f}%' for c in changes],
               textposition='auto', textfont=dict(color='white', size=12, family='Arial')),
        row=1, col=2
    )
    
    # Confidence levels
    fig1.add_trace(
        go.Scatter(x=tickers, y=confidences, mode='markers+lines',
                   name='Confidence %', marker_size=10,
                   marker_color='purple', line_color='purple'),
        row=2, col=1
    )
    
    # Price distribution
    fig1.add_trace(
        go.Histogram(x=changes, nbinsx=20, name='Change Distribution',
                     marker_color='orange', opacity=0.7),
        row=2, col=2
    )
    
    fig1.update_layout(
        height=800, 
        showlegend=True, 
        title_text="Comprehensive Prediction Analysis",
        font=dict(color='#333333', size=12, family='Arial'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Create performance summary chart
    performance_df = pd.DataFrame({
        'Ticker': tickers,
        'Current_Price': current_prices,
        'Predicted_Price': predicted_prices,
        'Change_%': changes,
        'Confidence_%': confidences
    })
    
    fig2 = px.scatter(performance_df, x='Change_%', y='Confidence_%', 
                      size='Current_Price', color='Change_%',
                      hover_data=['Ticker', 'Current_Price', 'Predicted_Price'],
                      title='Risk-Return Analysis: Expected Change vs Confidence',
                      color_continuous_scale='RdYlGn')
    fig2.add_hline(y=85, line_dash="dash", line_color="gray", 
                   annotation_text="High Confidence Threshold",
                   annotation_font=dict(color='#333333', size=10))
    fig2.add_vline(x=0, line_dash="dash", line_color="gray",
                   annotation_text="Neutral Change",
                   annotation_font=dict(color='#333333', size=10))
    
    fig2.update_layout(
        font=dict(color='#333333', size=12, family='Arial'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig1, fig2

def get_technical_indicators(ticker_symbol):
    """Get technical indicators for a stock"""
    try:
        import yfinance as yf
        import ta
        
        # Fetch data
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period="1y")
        
        if len(data) < 50:
            return None
            
        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(close=data['Close'], window=14)
        current_rsi = rsi.rsi().iloc[-1]
        
        # Calculate MACD
        macd = ta.trend.MACD(close=data['Close'])
        current_macd = macd.macd().iloc[-1]
        macd_signal = macd.macd_signal().iloc[-1]
        macd_diff = current_macd - macd_signal
        
        # Get 52-week high/low
        high_52w = data['High'].max()
        low_52w = data['Low'].min()
        current_price = data['Close'].iloc[-1]
        
        return {
            'rsi': current_rsi,
            'macd': current_macd,
            'macd_signal': macd_signal,
            'macd_diff': macd_diff,
            'high_52w': high_52w,
            'low_52w': low_52w,
            'current_price': current_price
        }
    except Exception as e:
        return None

def create_market_sentiment_gauge(avg_change, confidence):
    """Create market sentiment gauge"""
    # Determine sentiment color and level
    if avg_change > 2:
        sentiment = "Very Bullish"
        color = "green"
        value = 80 + min(avg_change, 20)
    elif avg_change > 0:
        sentiment = "Bullish"
        color = "lightgreen"
        value = 60 + avg_change * 10
    elif avg_change > -2:
        sentiment = "Bearish"
        color = "orange"
        value = 40 + avg_change * 10
    else:
        sentiment = "Very Bearish"
        color = "red"
        value = max(20 + avg_change, 0)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Market Sentiment - {sentiment}"},
        delta = {'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "lightgreen"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(
        height=400, 
        font=dict(color='#333333', size=16, family='Arial'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 StockSensei</h1>', unsafe_allow_html=True)
    
    # Live indicator
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.markdown(f'''
    <div style="text-align: center;">
        <span class="live-badge">🔴 LIVE</span>
        <span class="live-badge">🤖 AI-Powered</span>
        <span class="live-badge">📊 Real-time Data</span>
        <span class="live-badge">⏰ {current_time}</span>
    </div>
    ''', unsafe_allow_html=True)
    
    # Market status
    market_status = check_market_hours()
    market_badge = "🟢 OPEN" if market_status['is_open'] else "🔴 CLOSED"
    badge_color = "#28a745" if market_status['is_open'] else "#dc3545"
    st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 10px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="margin: 0; color: #1f2937;">US Market Status</h3>
        <p style="margin: 5px 0; font-size: 1.2em; font-weight: bold; color: {badge_color};">{market_badge}</p>
        <p style="margin: 0; font-size: 0.9em; color: #495057;">{market_status['current_time']} - {market_status['day']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown('<div class="sidebar-header">📋 Navigation</div>', unsafe_allow_html=True)
    app_mode = st.sidebar.radio(
        "Choose Mode:",
        ["🎯 Live Predictions", "📊 Market Overview", "💼 Portfolio Analysis", "📈 Historical Analysis"]
    )
    
    # Load predictors
    predictor = load_predictor()
    comprehensive_predictor = load_comprehensive_predictor()
    
    if predictor is None:
        st.error("❌ Cannot load prediction models. Please check model files.")
        st.stop()
    
    # Get available tickers
    all_tickers = get_available_tickers()
    
    if app_mode == "🎯 Live Predictions":
        st.header("🎯 Live Stock Predictions")
        
        # Prediction settings
        st.sidebar.markdown('<div class="sidebar-header">⚙️ Prediction Settings</div>', unsafe_allow_html=True)
        
        prediction_type = st.sidebar.radio(
            "Prediction Type:",
            ["Single Stock", "Multiple Stocks", "Custom Portfolio"]
        )
        
        days_ahead = st.sidebar.slider("Days Ahead:", 1, 5, 1)
        
        if prediction_type == "Single Stock":
            # Single stock prediction
            selected_ticker = st.sidebar.selectbox("Select Stock:", all_tickers)
            
            # Quick action buttons
            col1, col2, col3 = st.sidebar.columns(3)
            quick_tickers = ['AAPL', 'MSFT', 'TSLA']
            for i, ticker in enumerate(quick_tickers):
                if [col1, col2, col3][i].button(ticker):
                    selected_ticker = ticker
            
            if st.sidebar.button("🔮 Get Prediction", type="primary"):
                with st.spinner(f"Analyzing {selected_ticker}..."):
                    result = predictor.predict_future_price(selected_ticker, days_ahead)
                
                if 'error' not in result:
                    # Display prediction
                    change_pct = result['price_change_pct']
                    card_class = "bullish-card" if change_pct > 0 else "bearish-card"
                    direction = "📈" if change_pct > 0 else "📉"
                    
                    st.markdown(f'''
                    <div class="prediction-card {card_class}">
                        <h2>{direction} {selected_ticker} Live Prediction</h2>
                        <h3>Current: ${result['current_price']:.2f} → Predicted: ${result['predicted_price']:.2f}</h3>
                        <h3>Expected Change: {change_pct:+.2f}%</h3>
                        <p>📅 Prediction Date: {result['prediction_date']} ({days_ahead} day{'s' if days_ahead > 1 else ''} ahead)</p>
                        <p>🎯 Confidence: {result['confidence']:.1f}% | 📊 Volatility: {result['recent_volatility']:.3f}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Get technical indicators
                    tech_indicators = get_technical_indicators(selected_ticker)
                    
                    # Additional metrics
                    if tech_indicators:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            rsi_color = "#dc3545" if tech_indicators['rsi'] > 70 else "#28a745" if tech_indicators['rsi'] < 30 else "#ffc107"
                            st.markdown(f'''
                            <div class="metric-box">
                                <h3 style="color: {rsi_color}">{tech_indicators['rsi']:.1f}</h3>
                                <p>RSI (14)</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col2:
                            macd_color = "#28a745" if tech_indicators['macd_diff'] > 0 else "#dc3545"
                            st.markdown(f'''
                            <div class="metric-box">
                                <h3 style="color: {macd_color}">{tech_indicators['macd']:.2f}</h3>
                                <p>MACD</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f'''
                            <div class="metric-box">
                                <h3 style="color: #28a745">${tech_indicators['high_52w']:.2f}</h3>
                                <p>52 Week High</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f'''
                            <div class="metric-box">
                                <h3 style="color: #dc3545">${tech_indicators['low_52w']:.2f}</h3>
                                <p>52 Week Low</p>
                            </div>
                            ''', unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ Technical indicators not available")
                else:
                    st.error(f"❌ Prediction failed: {result['error']}")
        
        elif prediction_type == "Multiple Stocks":
            # Multiple stocks prediction
            selected_tickers = st.sidebar.multiselect(
                "Select Stocks:",
                all_tickers,
                default=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
            )
            
            if st.sidebar.button("🚀 Get Multiple Predictions", type="primary"):
                if selected_tickers:
                    with st.spinner(f"Analyzing {len(selected_tickers)} stocks..."):
                        results = predictor.get_multiple_predictions(selected_tickers, days_ahead)
                    
                    # Create visualizations
                    fig1, fig2 = create_prediction_visualization(results)
                    
                    if fig1 and fig2:
                        st.plotly_chart(fig1, use_container_width=True)
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Results table
                        st.subheader("📋 Detailed Results")
                        table_data = []
                        for ticker, result in results.items():
                            if 'error' not in result:
                                table_data.append({
                                    'Ticker': ticker,
                                    'Current Price': f"${result['current_price']:.2f}",
                                    'Predicted Price': f"${result['predicted_price']:.2f}",
                                    'Change': f"{result['price_change_pct']:+.2f}%",
                                    'Confidence': f"{result['confidence']:.1f}%",
                                    'Prediction Date': result['prediction_date']
                                })
                        
                        if table_data:
                            df = pd.DataFrame(table_data)
                            st.dataframe(df, use_container_width=True)
                else:
                    st.warning("Please select at least one stock.")
        
        else:  # Custom Portfolio
            st.sidebar.markdown("**Build Custom Portfolio:**")
            portfolio_tickers = st.sidebar.multiselect(
                "Add Stocks to Portfolio:",
                all_tickers,
                default=[]
            )
            
            if portfolio_tickers:
                # Portfolio weights
                st.sidebar.markdown("**Portfolio Weights:**")
                weights = {}
                remaining_weight = 100.0
                
                for i, ticker in enumerate(portfolio_tickers):
                    if i == len(portfolio_tickers) - 1:
                        # Last stock gets remaining weight
                        weights[ticker] = remaining_weight
                        st.sidebar.text(f"{ticker}: {remaining_weight:.1f}%")
                    else:
                        weight = st.sidebar.slider(f"{ticker}:", 0.0, remaining_weight, 
                                                 min(20.0, remaining_weight), step=0.5)
                        weights[ticker] = weight
                        remaining_weight -= weight
                
                if st.sidebar.button("📊 Analyze Portfolio", type="primary"):
                    with st.spinner("Analyzing portfolio..."):
                        results = predictor.get_multiple_predictions(portfolio_tickers, days_ahead)
                    
                    # Calculate portfolio metrics
                    portfolio_value = 0
                    portfolio_change = 0
                    valid_results = 0
                    
                    for ticker, result in results.items():
                        if 'error' not in result:
                            weight = weights[ticker] / 100
                            portfolio_change += result['price_change_pct'] * weight
                            valid_results += 1
                    
                    if valid_results > 0:
                        st.subheader("💼 Portfolio Analysis")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Portfolio Expected Change", f"{portfolio_change:+.2f}%")
                        with col2:
                            st.metric("Number of Stocks", str(valid_results))
                        with col3:
                            avg_confidence = np.mean([r['confidence'] for r in results.values() if 'error' not in r])
                            st.metric("Average Confidence", f"{avg_confidence:.1f}%")
                        
                        # Portfolio composition chart
                        fig = px.pie(values=list(weights.values()), names=list(weights.keys()),
                                   title="Portfolio Composition")
                        st.plotly_chart(fig, use_container_width=True)
    
    elif app_mode == "📊 Market Overview":
        st.header("📊 Market Overview & Sentiment")
        
        # Display major indices
        st.subheader("📈 Major Market Indices")
        
        # Fetch market indices data
        try:
            import yfinance as yf
            
            # Define major indices
            indices = {
                '^GSPC': 'S&P 500',
                '^DJI': 'Dow Jones',
                '^IXIC': 'NASDAQ',
                '^RUT': 'Russell 2000'
            }
            
            col1, col2, col3, col4 = st.columns(4)
            cols = [col1, col2, col3, col4]
            
            for i, (symbol, name) in enumerate(indices.items()):
                ticker = yf.Ticker(symbol)
                info = ticker.history(period='2d')
                if len(info) >= 2:
                    current_price = info['Close'].iloc[-1]
                    prev_price = info['Close'].iloc[-2]
                    change = ((current_price - prev_price) / prev_price) * 100
                    
                    with cols[i]:
                        delta_color = "normal" if symbol != '^VIX' else "inverse"
                        st.metric(
                            label=name,
                            value=f"{current_price:,.2f}",
                            delta=f"{change:.2f}%",
                            delta_color=delta_color
                        )
        except Exception as e:
            st.warning("Could not fetch market indices data")
        
        # Sector Performance
        st.subheader("🎯 Sector Performance")
        try:
            sectors = {
                'XLK': 'Technology',
                'XLF': 'Financials', 
                'XLV': 'Healthcare',
                'XLE': 'Energy',
                'XLI': 'Industrials',
                'XLY': 'Consumer Disc.',
                'XLP': 'Consumer Staples',
                'XLRE': 'Real Estate'
            }
            
            sector_data = []
            for symbol, name in sectors.items():
                ticker = yf.Ticker(symbol)
                info = ticker.history(period='2d')
                if len(info) >= 2:
                    current_price = info['Close'].iloc[-1]
                    prev_price = info['Close'].iloc[-2]
                    change = ((current_price - prev_price) / prev_price) * 100
                    sector_data.append({
                        'Sector': name,
                        'Change %': change,
                        'Price': current_price
                    })
            
            if sector_data:
                sector_df = pd.DataFrame(sector_data).sort_values('Change %', ascending=False)
                
                # Create sector performance chart
                fig = px.bar(sector_df, x='Sector', y='Change %', 
                            color='Change %',
                            color_continuous_scale='RdYlGn',
                            title='Sector Performance (Daily Change %)')
                fig.update_layout(
                    height=400,
                    font=dict(color='#333333', size=12, family='Arial'),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Sector data unavailable")
        
        st.divider()
        
        if st.button("🌍 Get Live Market Overview", type="primary"):
            # Use top 20 stocks for market overview
            market_tickers = all_tickers[:20]
            
            with st.spinner("Analyzing market conditions..."):
                results = predictor.get_multiple_predictions(market_tickers, 1)
            
            # Calculate market metrics
            successful_results = [r for r in results.values() if 'error' not in r]
            
            if successful_results:
                changes = [r['price_change_pct'] for r in successful_results]
                confidences = [r['confidence'] for r in successful_results]
                
                avg_change = np.mean(changes)
                avg_confidence = np.mean(confidences)
                bullish_count = sum(1 for c in changes if c > 0)
                bearish_count = len(changes) - bullish_count
                
                # Market sentiment gauge
                sentiment_fig = create_market_sentiment_gauge(avg_change, avg_confidence)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.plotly_chart(sentiment_fig, use_container_width=True)
                
                with col2:
                    # Market statistics and news
                    st.subheader("📊 Market Statistics & News")
                    
                    # Fetch latest news (placeholder)
                    latest_news = [
                        "The Fed raises interest rates by 25 basis points.",
                        "Tech stocks soar as Nasdaq hits new high.",
                        "Oil prices dip slightly as supply chain stabilizes."
                    ]
                    
                    st.write("**📅 Latest News**")
                    for news_item in latest_news:
                        st.write(f"- {news_item}")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Average Change", f"{avg_change:+.2f}%")
                        st.metric("Bullish Stocks", f"{bullish_count}/{len(successful_results)}")
                    
                    with col_b:
                        st.metric("Average Confidence", f"{avg_confidence:.1f}%")
                        st.metric("Bearish Stocks", f"{bearish_count}/{len(successful_results)}")
                    
                    with col_c:
                        st.metric("Volatility", f"{np.std(changes):.2f}%")
                        st.metric("Range", f"{np.min(changes):.1f}% to {np.max(changes):.1f}%")
                
                # Top movers
                sorted_results = sorted([(k, v) for k, v in results.items() if 'error' not in v],
                                      key=lambda x: x[1]['price_change_pct'], reverse=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🚀 Top Gainers")
                    for i, (ticker, result) in enumerate(sorted_results[:5]):
                        st.write(f"{i+1}. **{ticker}**: {result['price_change_pct']:+.2f}% "
                               f"(Confidence: {result['confidence']:.1f}%)")
                
                with col2:
                    st.subheader("📉 Top Losers")
                    for i, (ticker, result) in enumerate(sorted_results[-5:]):
                        st.write(f"{i+1}. **{ticker}**: {result['price_change_pct']:+.2f}% "
                               f"(Confidence: {result['confidence']:.1f}%)")
    
    elif app_mode == "💼 Portfolio Analysis":
        st.header("💼 Portfolio Analysis & Optimization")
        st.info("🚧 Advanced portfolio analysis features coming soon!")
        
        # Basic portfolio analysis
        st.subheader("📊 Quick Portfolio Check")
        portfolio_tickers = st.multiselect(
            "Select stocks for portfolio analysis:",
            all_tickers,
            default=['AAPL', 'MSFT', 'GOOGL']
        )
        
        if portfolio_tickers and st.button("Analyze Portfolio"):
            with st.spinner("Analyzing portfolio..."):
                results = predictor.get_multiple_predictions(portfolio_tickers, 1)
            
            # Portfolio metrics
            valid_results = [r for r in results.values() if 'error' not in r]
            if valid_results:
                changes = [r['price_change_pct'] for r in valid_results]
                portfolio_return = np.mean(changes)
                portfolio_risk = np.std(changes)
                sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Return", f"{portfolio_return:+.2f}%")
                with col2:
                    st.metric("Portfolio Risk", f"{portfolio_risk:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    elif app_mode == "📈 Historical Analysis":
        st.header("📈 Historical Analysis & Technical Indicators")
        
        # Historical analysis settings
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_ticker = st.selectbox("Select stock for historical analysis:", all_tickers)
        with col2:
            period = st.selectbox("Time Period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
        
        try:
            # Fetch historical data with technical indicators
            import yfinance as yf
            import ta
            
            ticker = yf.Ticker(selected_ticker)
            hist_data = ticker.history(period=period)
            
            if len(hist_data) > 0:
                # Calculate technical indicators
                hist_data['RSI'] = ta.momentum.RSIIndicator(close=hist_data['Close'], window=14).rsi()
                hist_data['MACD'] = ta.trend.MACD(close=hist_data['Close']).macd()
                hist_data['MACD_Signal'] = ta.trend.MACD(close=hist_data['Close']).macd_signal()
                hist_data['BB_Upper'] = ta.volatility.BollingerBands(close=hist_data['Close']).bollinger_hband()
                hist_data['BB_Lower'] = ta.volatility.BollingerBands(close=hist_data['Close']).bollinger_lband()
                hist_data['SMA_20'] = ta.trend.SMAIndicator(close=hist_data['Close'], window=20).sma_indicator()
                hist_data['SMA_50'] = ta.trend.SMAIndicator(close=hist_data['Close'], window=50).sma_indicator()
                
                # Create technical analysis charts
                st.subheader(f"📈 {selected_ticker} Technical Analysis")
                
                # Price and volume chart
                fig = make_subplots(rows=3, cols=1, 
                                   shared_xaxes=True,
                                   vertical_spacing=0.03,
                                   row_heights=[0.5, 0.25, 0.25],
                                   subplot_titles=[f'{selected_ticker} Price & Moving Averages', 
                                                 'RSI (14)', 
                                                 'MACD'])
                
                # Price chart with moving averages
                fig.add_trace(go.Candlestick(x=hist_data.index,
                                            open=hist_data['Open'],
                                            high=hist_data['High'],
                                            low=hist_data['Low'],
                                            close=hist_data['Close'],
                                            name='Price'), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['SMA_20'],
                                       name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['SMA_50'],
                                       name='SMA 50', line=dict(color='blue', width=1)), row=1, col=1)
                
                # RSI chart
                fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['RSI'],
                                       name='RSI', line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD chart
                fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['MACD'],
                                       name='MACD', line=dict(color='blue')), row=3, col=1)
                fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['MACD_Signal'],
                                       name='Signal', line=dict(color='red')), row=3, col=1)
                
                fig.update_layout(height=800, showlegend=True,
                                font=dict(color='#333333', size=12, family='Arial'),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)')
                fig.update_xaxes(rangeslider_visible=False)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Key metrics
                tech_indicators = get_technical_indicators(selected_ticker)
                if tech_indicators:
                    st.subheader("📋 Current Technical Indicators")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        rsi_status = "Overbought" if tech_indicators['rsi'] > 70 else "Oversold" if tech_indicators['rsi'] < 30 else "Neutral"
                        st.metric("RSI (14)", f"{tech_indicators['rsi']:.1f}", rsi_status)
                    
                    with col2:
                        macd_status = "Bullish" if tech_indicators['macd_diff'] > 0 else "Bearish"
                        st.metric("MACD", f"{tech_indicators['macd']:.2f}", macd_status)
                    
                    with col3:
                        price_vs_high = ((tech_indicators['current_price'] - tech_indicators['high_52w']) / tech_indicators['high_52w']) * 100
                        st.metric("52W High", f"${tech_indicators['high_52w']:.2f}", f"{price_vs_high:.1f}%")
                    
                    with col4:
                        price_vs_low = ((tech_indicators['current_price'] - tech_indicators['low_52w']) / tech_indicators['low_52w']) * 100
                        st.metric("52W Low", f"${tech_indicators['low_52w']:.2f}", f"+{price_vs_low:.1f}%")
            else:
                st.error("No historical data available for this ticker")
        
        except Exception as e:
            st.error(f"Could not load historical data: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0;">
        <p style="color: #6b7280; font-size: 0.9rem; margin: 0;">
            <strong>© Copyright Punit Kumar</strong> | 
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
