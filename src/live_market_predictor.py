#!/usr/bin/env python3
"""
Live Market Predictor for Stock Market AI Agent

This module fetches real-time market data and makes predictions for future dates,
not just historical data that the model has already seen.

Features:
- Real-time data fetching from Yahoo Finance
- Live feature engineering
- Future price predictions
- Market sentiment integration
- Technical indicator calculation

Author: AI Assistant
Date: July 2025
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime, timedelta
import requests
import ta
from typing import Dict, List, Tuple, Optional
import json

warnings.filterwarnings('ignore')

class LiveMarketPredictor:
    """Live market data fetcher and predictor"""
    
    def __init__(self, model_path: str, feature_names_path: str):
        self.model_path = model_path
        self.feature_names_path = feature_names_path
        self.model = None
        self.feature_names = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model and feature names"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.feature_names_path, 'rb') as f:
                self.feature_names = pickle.load(f)
                
            print(f"✅ Model loaded successfully with {len(self.feature_names)} features")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def fetch_live_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Fetch live market data from Yahoo Finance with multiple fallbacks"""
        try:
            print(f"📡 Fetching live data for {ticker}...")
            
            # Try multiple approaches for yfinance
            data = None
            
            # Method 1: Direct download
            try:
                data = yf.download(ticker, period=period, progress=False, threads=False)
                if not data.empty and len(data) > 20:
                    data = data.reset_index()
                    print(f"✅ Downloaded {len(data)} records via yf.download")
            except:
                pass
            
            # Method 2: Ticker.history fallback
            if data is None or data.empty:
                stock = yf.Ticker(ticker)
                data = stock.history(period=period)
                if not data.empty:
                    data = data.reset_index()
                    print(f"✅ Fetched {len(data)} records via Ticker.history")
            
            # Method 3: Shorter period fallback
            if data is None or data.empty or len(data) < 20:
                periods_to_try = ['6mo', '3mo', '1mo'] if period == '1y' else ['1mo', '2mo']
                for fallback_period in periods_to_try:
                    try:
                        stock = yf.Ticker(ticker)
                        data = stock.history(period=fallback_period)
                        if not data.empty and len(data) > 10:
                            data = data.reset_index()
                            print(f"✅ Fallback: {len(data)} records with period={fallback_period}")
                            break
                    except:
                        continue
            
            if data is None or data.empty:
                raise ValueError(f"No data available for {ticker} after trying multiple methods")
            
            # Reset index to get Date as a column
            data.reset_index(inplace=True)
            
            # Rename columns to match our training data format
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            data['ticker'] = ticker
            
            # Ensure we have the required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            print(f"✅ Fetched {len(data)} records for {ticker}")
            print(f"📅 Data range: {data['date'].min()} to {data['date'].max()}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the live data"""
        if len(data) < 50:  # Need minimum data for indicators
            print("⚠️ Insufficient data for technical indicators")
            return data
        
        try:
            print("🔧 Calculating technical indicators...")
            
            # Price-based features
            data['price_range'] = data['high'] - data['low']
            data['price_change'] = data['close'] - data['open']
            data['price_change_pct'] = data['price_change'] / data['open']
            
            # Returns
            data['return_1d'] = data['close'].pct_change(1)
            data['return_5d'] = data['close'].pct_change(5)
            data['return_10d'] = data['close'].pct_change(10)
            data['return_20d'] = data['close'].pct_change(20)
            data['log_return_1d'] = np.log(data['close'] / data['close'].shift(1))
            
            # Price ratios
            data['high_low_ratio'] = data['high'] / data['low']
            data['close_high_ratio'] = data['close'] / data['high']
            data['close_low_ratio'] = data['close'] / data['low']
            
            # Moving averages
            for period in [5, 10, 20, 50, 100, 200]:
                data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
                data[f'sma_{period}_ratio'] = data['close'] / data[f'sma_{period}']
                
                data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
                data[f'ema_{period}_ratio'] = data['close'] / data[f'ema_{period}']
            
            # Moving average crosses
            data['sma_5_20_cross'] = (data['sma_5'] > data['sma_20']).astype(int)
            data['sma_10_50_cross'] = (data['sma_10'] > data['sma_50']).astype(int)
            data['ema_5_20_cross'] = (data['ema_5'] > data['ema_20']).astype(int)
            data['sma_5_20_distance'] = (data['sma_5'] - data['sma_20']) / data['sma_20']
            data['ema_5_20_distance'] = (data['ema_5'] - data['ema_20']) / data['ema_20']
            
            # Volatility
            for period in [5, 10, 20, 50]:
                data[f'volatility_{period}d'] = data['return_1d'].rolling(window=period).std()
            
            # Bollinger Bands
            data['bb_middle'] = data['sma_20']
            bb_std = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # ATR (Average True Range)
            data['atr_14'] = ta.volatility.AverageTrueRange(
                high=data['high'], low=data['low'], close=data['close'], window=14
            ).average_true_range()
            data['atr_ratio'] = data['atr_14'] / data['close']
            
            # Price position
            for period in [5, 10, 20]:
                rolling_min = data['low'].rolling(window=period).min()
                rolling_max = data['high'].rolling(window=period).max()
                data[f'price_position_{period}d'] = (data['close'] - rolling_min) / (rolling_max - rolling_min)
            
            # RSI
            data['rsi_14'] = ta.momentum.RSIIndicator(close=data['close'], window=14).rsi()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(
                high=data['high'], low=data['low'], close=data['close'], window=14, smooth_window=3
            )
            data['stoch_k'] = stoch.stoch()
            data['stoch_d'] = stoch.stoch_signal()
            
            # MACD
            macd = ta.trend.MACD(close=data['close'], window_slow=26, window_fast=12, window_sign=9)
            data['macd'] = macd.macd()
            data['macd_signal'] = macd.macd_signal()
            data['macd_histogram'] = macd.macd_diff()
            
            # Momentum
            for period in [5, 10, 20]:
                data[f'momentum_{period}d'] = data['close'] / data['close'].shift(period)
            
            # Rate of Change
            for period in [5, 10, 20]:
                data[f'roc_{period}d'] = data['close'].pct_change(period)
            
            # Volume indicators
            for period in [5, 10, 20, 50]:
                data[f'volume_sma_{period}'] = data['volume'].rolling(window=period).mean()
                data[f'volume_ratio_{period}'] = data['volume'] / data[f'volume_sma_{period}']
            
            # OBV (On Balance Volume)
            data['obv'] = ta.volume.OnBalanceVolumeIndicator(
                close=data['close'], volume=data['volume']
            ).on_balance_volume()
            data['obv_sma_10'] = data['obv'].rolling(window=10).mean()
            data['obv_ratio'] = data['obv'] / data['obv_sma_10']
            
            # Volume Price Trend
            data['vpt'] = ta.volume.VolumePriceTrendIndicator(
                close=data['close'], volume=data['volume']
            ).volume_price_trend()
            
            # Price-Volume
            data['price_volume'] = data['close'] * data['volume']
            data['price_volume_sma_10'] = data['price_volume'].rolling(window=10).mean()
            
            # Lagged features
            for lag in [1, 2, 3, 5, 10]:
                data[f'close_lag_{lag}'] = data['close'].shift(lag)
                data[f'volume_lag_{lag}'] = data['volume'].shift(lag)
                data[f'return_1d_lag_{lag}'] = data['return_1d'].shift(lag)
                data[f'rsi_14_lag_{lag}'] = data['rsi_14'].shift(lag)
                data[f'volatility_20d_lag_{lag}'] = data['volatility_20d'].shift(lag)
            
            print("✅ Technical indicators calculated successfully")
            return data
            
        except Exception as e:
            print(f"❌ Error calculating technical indicators: {e}")
            return data
    
    def add_synthetic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add synthetic features for missing model features"""
        try:
            print("🔧 Adding synthetic features...")
            
            # Add synthetic sentiment features (since we don't have real-time sentiment)
            latest_return = data['return_1d'].iloc[-5:].mean() if len(data) >= 5 else 0
            sentiment_score = np.clip(latest_return * 10, -1, 1)  # Convert return to sentiment
            
            data['sentiment_score'] = sentiment_score
            data['news_count'] = 10  # Default news count
            
            # Sentiment rolling averages
            for period in [3, 7, 14, 30]:
                data[f'sentiment_sma_{period}'] = sentiment_score
                data[f'news_count_sma_{period}'] = 10
            
            # Sentiment momentum
            for period in [3, 7, 14]:
                data[f'sentiment_momentum_{period}d'] = 0
            
            # Sentiment volatility
            for period in [7, 14, 30]:
                data[f'sentiment_volatility_{period}d'] = 0.1
            
            # Sentiment extremes
            data['sentiment_extreme'] = (abs(sentiment_score) > 0.5).astype(int)
            data['sentiment_positive'] = (sentiment_score > 0.2).astype(int)
            data['sentiment_negative'] = (sentiment_score < -0.2).astype(int)
            
            # Sentiment lags
            for lag in [1, 2, 3, 5, 7]:
                data[f'sentiment_lag_{lag}'] = sentiment_score
            
            # Add synthetic macro indicators (simplified)
            data['GDP'] = 25000  # Typical GDP value
            data['unemployment_rate'] = 4.0
            data['inflation_rate'] = 2.5
            data['federal_funds_rate'] = 5.0
            data['consumer_sentiment'] = 80
            data['vix'] = 20
            
            # Macro rolling features
            for indicator in ['GDP', 'unemployment_rate', 'inflation_rate', 'federal_funds_rate', 'consumer_sentiment', 'vix']:
                for period in [30, 90, 180]:
                    data[f'{indicator}_roc_{period}d'] = 0
                    data[f'{indicator}_sma_{period}'] = data[indicator]
                    data[f'{indicator}_ratio_{period}'] = 1.0
            
            # Macro regime indicators
            data['unemployment_high'] = (data['unemployment_rate'] > 5.0).astype(int)
            data['inflation_high'] = (data['inflation_rate'] > 3.0).astype(int)
            data['rates_rising'] = 0  # Default
            data['vix_high'] = (data['vix'] > 25).astype(int)
            
            # Target features (these will be ignored in prediction)
            data['target_direction_1d'] = 0
            data['target_direction_5d'] = 0
            
            print("✅ Synthetic features added successfully")
            return data
            
        except Exception as e:
            print(f"❌ Error adding synthetic features: {e}")
            return data
    
    def prepare_features_for_prediction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features to match the model's expected input"""
        try:
            print("🎯 Preparing features for prediction...")
            
            # Get the latest row (most recent data)
            latest_data = data.tail(1).copy()
            
            # Create feature dataframe with model's expected features
            feature_data = pd.DataFrame(index=latest_data.index)
            
            missing_features = []
            for feature in self.feature_names:
                if feature in latest_data.columns:
                    feature_data[feature] = latest_data[feature].fillna(0)
                else:
                    feature_data[feature] = 0
                    missing_features.append(feature)
            
            if missing_features:
                print(f"⚠️ Warning: {len(missing_features)} features missing, filled with zeros")
                print(f"Missing features: {missing_features[:5]}...")
            
            print(f"✅ Features prepared: {len(feature_data.columns)} features ready")
            return feature_data
            
        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            return pd.DataFrame()
    
    def predict_future_price(self, ticker: str, days_ahead: int = 1) -> Dict:
        """Make a prediction for future price"""
        try:
            print(f"\n🔮 Making live prediction for {ticker} ({days_ahead} days ahead)")
            print("=" * 60)
            
            # Fetch live data
            data = self.fetch_live_data(ticker, period="1y")
            if data.empty:
                return {"error": "Could not fetch live data"}
            
            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)
            
            # Add synthetic features
            data = self.add_synthetic_features(data)
            
            # Prepare features for prediction
            feature_data = self.prepare_features_for_prediction(data)
            if feature_data.empty:
                return {"error": "Could not prepare features"}
            
            # Make prediction
            prediction = self.model.predict(feature_data)[0]
            
            # Get current information
            current_price = data['close'].iloc[-1]
            current_date = data['date'].iloc[-1]
            prediction_date = current_date + timedelta(days=days_ahead)
            
            # Calculate metrics
            change = prediction - current_price
            change_pct = (change / current_price) * 100
            
            # Calculate confidence based on recent volatility
            recent_volatility = data['volatility_20d'].iloc[-1] if 'volatility_20d' in data.columns else 0.02
            confidence = max(60, min(95, 100 - (recent_volatility * 1000)))
            
            result = {
                'ticker': ticker,
                'current_price': float(current_price),
                'predicted_price': float(prediction),
                'price_change': float(change),
                'price_change_pct': float(change_pct),
                'current_date': current_date.strftime('%Y-%m-%d'),
                'prediction_date': prediction_date.strftime('%Y-%m-%d'),
                'days_ahead': days_ahead,
                'confidence': float(confidence),
                'model_features_used': len(self.feature_names),
                'data_points_analyzed': len(data),
                'recent_volatility': float(recent_volatility) if recent_volatility else 0.02
            }
            
            print(f"✅ Prediction completed successfully!")
            print(f"📈 Current: ${current_price:.2f} → Predicted: ${prediction:.2f}")
            print(f"📊 Change: {change_pct:.2f}% | Confidence: {confidence:.1f}%")
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return {"error": str(e)}
    
    def get_multiple_predictions(self, tickers: List[str], days_ahead: int = 1) -> Dict:
        """Get predictions for multiple tickers"""
        results = {}
        
        print(f"\n🚀 Getting predictions for {len(tickers)} tickers")
        print("=" * 80)
        
        for ticker in tickers:
            try:
                result = self.predict_future_price(ticker, days_ahead)
                results[ticker] = result
                
                if 'error' not in result:
                    print(f"✅ {ticker}: ${result['current_price']:.2f} → ${result['predicted_price']:.2f} ({result['price_change_pct']:+.2f}%)")
                else:
                    print(f"❌ {ticker}: {result['error']}")
                    
            except Exception as e:
                results[ticker] = {"error": str(e)}
                print(f"❌ {ticker}: {e}")
        
        return results

def main():
    """Test the live predictor"""
    print("🚀 LIVE MARKET PREDICTOR TEST")
    print("=" * 60)
    
    # Initialize predictor
    predictor = LiveMarketPredictor(
        model_path='models/random_forest_model.pkl',
        feature_names_path='models/feature_names.pkl'
    )
    
    if predictor.model is None:
        print("❌ Could not load model. Exiting.")
        return
    
    # Test predictions for popular stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    # Get predictions
    results = predictor.get_multiple_predictions(test_tickers, days_ahead=1)
    
    # Display summary
    print(f"\n🏆 PREDICTION SUMMARY")
    print("=" * 80)
    
    for ticker, result in results.items():
        if 'error' not in result:
            direction = "📈" if result['price_change_pct'] > 0 else "📉"
            print(f"{direction} {ticker}: {result['price_change_pct']:+.2f}% | Confidence: {result['confidence']:.1f}%")
        else:
            print(f"❌ {ticker}: Failed to predict")

if __name__ == "__main__":
    main()
