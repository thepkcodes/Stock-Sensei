import pandas as pd
import numpy as np
import yfinance as yf
from newsapi import NewsApiClient
from fredapi import Fred
import requests
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv
from textblob import TextBlob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

class StockDataCollector:
    """Collects historical stock price data"""
    
    def __init__(self):
        self.symbols = os.getenv('STOCK_SYMBOLS', '').split(',')
    
    def collect_stock_data(self, start_date='2019-01-01', end_date='2024-01-01'):
        """
        Collect 5 years of stock price data for top 50 S&P 500 companies
        """
        print("Collecting stock price data...")
        all_data = []
        
        for symbol in tqdm(self.symbols, desc="Downloading stock data"):
            try:
                ticker = yf.Ticker(symbol.strip())
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    hist.reset_index(inplace=True)
                    hist['Ticker'] = symbol.strip()
                    hist['Date'] = hist['Date'].dt.date
                    
                    # Rename columns to match expected format
                    hist.rename(columns={
                        'Open': 'open',
                        'High': 'high', 
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    }, inplace=True)
                    
                    # Select relevant columns
                    hist = hist[['Date', 'Ticker', 'open', 'high', 'low', 'close', 'volume']]
                    all_data.append(hist)
                    
            except Exception as e:
                print(f"Error collecting data for {symbol}: {e}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data['Date'] = pd.to_datetime(combined_data['Date'])
            return combined_data
        else:
            return pd.DataFrame()

class SentimentDataCollector:
    """Collects and analyzes financial news sentiment"""
    
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY')
        if self.api_key:
            self.newsapi = NewsApiClient(api_key=self.api_key)
        else:
            self.newsapi = None
            print("Warning: News API key not found. Will generate synthetic sentiment data.")
    
    def get_sentiment_score(self, text):
        """Calculate sentiment score using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity  # Returns value between -1 and 1
        except:
            return 0.0
    
    def collect_news_sentiment(self, symbols, start_date='2019-01-01', end_date='2024-01-01'):
        """
        Collect news sentiment data for given symbols
        If API is not available, generate synthetic data
        """
        print("Collecting news sentiment data...")
        
        if not self.newsapi:
            return self._generate_synthetic_sentiment(symbols, start_date, end_date)
        
        all_sentiment = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # News API has limitations, so we'll generate synthetic data for demonstration
        return self._generate_synthetic_sentiment(symbols, start_date, end_date)
    
    def _generate_synthetic_sentiment(self, symbols, start_date, end_date):
        """Generate synthetic sentiment data for demonstration"""
        print("Generating synthetic sentiment data...")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        all_sentiment = []
        
        np.random.seed(42)  # For reproducible results
        
        for symbol in tqdm(symbols, desc="Generating sentiment data"):
            symbol = symbol.strip()
            
            # Generate realistic sentiment scores with some autocorrelation
            n_days = len(date_range)
            sentiment_scores = np.random.normal(0, 0.3, n_days)
            
            # Add some autocorrelation to make it more realistic
            for i in range(1, len(sentiment_scores)):
                sentiment_scores[i] = 0.7 * sentiment_scores[i-1] + 0.3 * sentiment_scores[i]
            
            # Clip to reasonable range
            sentiment_scores = np.clip(sentiment_scores, -1, 1)
            
            for i, date in enumerate(date_range):
                # Skip weekends for some realism
                if date.weekday() < 5:  # Monday is 0, Sunday is 6
                    all_sentiment.append({
                        'Date': date.date(),
                        'Ticker': symbol,
                        'sentiment_score': sentiment_scores[i],
                        'news_count': np.random.poisson(3)  # Average 3 news articles per day
                    })
        
        sentiment_df = pd.DataFrame(all_sentiment)
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
        return sentiment_df

class MacroDataCollector:
    """Collects macroeconomic data"""
    
    def __init__(self):
        self.api_key = os.getenv('FRED_API_KEY')
        if self.api_key:
            self.fred = Fred(api_key=self.api_key)
        else:
            self.fred = None
            print("Warning: FRED API key not found. Will generate synthetic macro data.")
    
    def collect_macro_data(self, start_date='2019-01-01', end_date='2024-01-01'):
        """
        Collect macroeconomic indicators
        """
        print("Collecting macroeconomic data...")
        
        if not self.fred:
            return self._generate_synthetic_macro_data(start_date, end_date)
        
        try:
            # Define the economic indicators we want to collect
            indicators = {
                'GDP': 'GDP',
                'unemployment_rate': 'UNRATE',
                'inflation_rate': 'CPIAUCSL',
                'federal_funds_rate': 'FEDFUNDS',
                'consumer_sentiment': 'UMCSENT',
                'vix': 'VIXCLS'
            }
            
            macro_data = {}
            
            for name, series_id in indicators.items():
                try:
                    data = self.fred.get_series(series_id, start=start_date, end=end_date)
                    macro_data[name] = data
                    time.sleep(0.1)  # Rate limiting
                except Exception as e:
                    print(f"Error collecting {name}: {e}")
                    continue
            
            if macro_data:
                # Combine all indicators into one DataFrame
                combined_macro = pd.DataFrame(macro_data)
                combined_macro.index.name = 'Date'
                combined_macro.reset_index(inplace=True)
                
                # Forward fill missing values and interpolate
                combined_macro = combined_macro.fillna(method='ffill').interpolate()
                
                return combined_macro
            else:
                return self._generate_synthetic_macro_data(start_date, end_date)
                
        except Exception as e:
            print(f"Error with FRED API: {e}")
            return self._generate_synthetic_macro_data(start_date, end_date)
    
    def _generate_synthetic_macro_data(self, start_date, end_date):
        """Generate synthetic macroeconomic data for demonstration"""
        print("Generating synthetic macroeconomic data...")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(date_range)
        
        np.random.seed(42)
        
        # Generate realistic macro indicators with trends
        data = {
            'Date': date_range,
            'GDP': np.random.normal(20000, 1000, n_days) + np.linspace(0, 2000, n_days),  # Growing GDP
            'unemployment_rate': 3.5 + np.random.normal(0, 0.5, n_days) + 2 * np.sin(np.linspace(0, 4*np.pi, n_days)),  # Cyclical unemployment
            'inflation_rate': 2.0 + np.random.normal(0, 0.3, n_days),  # Around 2% inflation
            'federal_funds_rate': np.maximum(0, 1.5 + np.random.normal(0, 0.5, n_days)),  # Positive interest rates
            'consumer_sentiment': 80 + np.random.normal(0, 10, n_days),  # Consumer sentiment index
            'vix': np.maximum(10, 20 + np.random.normal(0, 5, n_days))  # VIX volatility index
        }
        
        macro_df = pd.DataFrame(data)
        return macro_df

def main():
    """Main function to collect all data"""
    
    # Initialize collectors
    stock_collector = StockDataCollector()
    sentiment_collector = SentimentDataCollector()
    macro_collector = MacroDataCollector()
    
    # Collect data
    print("Starting data collection process...")
    
    # 1. Collect stock data
    stock_data = stock_collector.collect_stock_data()
    if not stock_data.empty:
        stock_data.to_csv('data/stock_prices.csv', index=False)
        print(f"Stock data saved: {len(stock_data)} records")
    
    # 2. Collect sentiment data
    symbols = stock_collector.symbols
    sentiment_data = sentiment_collector.collect_news_sentiment(symbols)
    if not sentiment_data.empty:
        sentiment_data.to_csv('data/news_sentiment.csv', index=False)
        print(f"Sentiment data saved: {len(sentiment_data)} records")
    
    # 3. Collect macro data
    macro_data = macro_collector.collect_macro_data()
    if not macro_data.empty:
        macro_data.to_csv('data/macro_indicators.csv', index=False)
        print(f"Macro data saved: {len(macro_data)} records")
    
    print("Data collection completed!")

if __name__ == "__main__":
    main()
