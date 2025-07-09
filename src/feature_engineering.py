import pandas as pd
import numpy as np
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import warnings
warnings.filterwarnings('ignore')

class StockFeatureEngineer:
    """Feature engineering class for stock market data"""
    
    def __init__(self):
        self.features_created = []
    
    def create_price_features(self, df):
        """Create price-based features"""
        df = df.copy()
        
        # Basic price features
        df['price_range'] = df['high'] - df['low']
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
        
        # Returns
        df['return_1d'] = df['close'].pct_change()
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)
        
        # Log returns
        df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
        
        # High-Low ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_high_ratio'] = df['close'] / df['high']
        df['close_low_ratio'] = df['close'] / df['low']
        
        self.features_created.extend([
            'price_range', 'price_change', 'price_change_pct',
            'return_1d', 'return_5d', 'return_10d', 'return_20d',
            'log_return_1d', 'high_low_ratio', 'close_high_ratio', 'close_low_ratio'
        ])
        
        return df
    
    def create_moving_averages(self, df):
        """Create moving average features"""
        df = df.copy()
        
        # Simple Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            
        # Exponential Moving Averages
        for window in [5, 10, 20, 50]:
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']
        
        # Moving average crossovers
        df['sma_5_20_cross'] = np.where(df['sma_5'] > df['sma_20'], 1, 0)
        df['sma_10_50_cross'] = np.where(df['sma_10'] > df['sma_50'], 1, 0)
        df['ema_5_20_cross'] = np.where(df['ema_5'] > df['ema_20'], 1, 0)
        
        # Moving average distances
        df['sma_5_20_distance'] = (df['sma_5'] - df['sma_20']) / df['sma_20'] * 100
        df['ema_5_20_distance'] = (df['ema_5'] - df['ema_20']) / df['ema_20'] * 100
        
        self.features_created.extend([
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
            'sma_5_ratio', 'sma_10_ratio', 'sma_20_ratio', 'sma_50_ratio', 'sma_100_ratio', 'sma_200_ratio',
            'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'ema_5_ratio', 'ema_10_ratio', 'ema_20_ratio', 'ema_50_ratio',
            'sma_5_20_cross', 'sma_10_50_cross', 'ema_5_20_cross',
            'sma_5_20_distance', 'ema_5_20_distance'
        ])
        
        return df
    
    def create_volatility_features(self, df):
        """Create volatility-based features"""
        df = df.copy()
        
        # Rolling volatility (standard deviation of returns)
        for window in [5, 10, 20, 50]:
            df[f'volatility_{window}d'] = df['return_1d'].rolling(window=window).std() * np.sqrt(252)
        
        # Bollinger Bands
        bb_20 = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb_20.bollinger_hband()
        df['bb_lower'] = bb_20.bollinger_lband()
        df['bb_middle'] = bb_20.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Average True Range
        atr_14 = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr_14'] = atr_14.average_true_range()
        df['atr_ratio'] = df['atr_14'] / df['close']
        
        # Price position in range
        for window in [5, 10, 20]:
            df[f'price_position_{window}d'] = (df['close'] - df['low'].rolling(window).min()) / \
                                             (df['high'].rolling(window).max() - df['low'].rolling(window).min())
        
        self.features_created.extend([
            'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_50d',
            'bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position',
            'atr_14', 'atr_ratio',
            'price_position_5d', 'price_position_10d', 'price_position_20d'
        ])
        
        return df
    
    def create_momentum_features(self, df):
        """Create momentum-based features"""
        df = df.copy()
        
        # RSI
        rsi_14 = RSIIndicator(close=df['close'], window=14)
        df['rsi_14'] = rsi_14.rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Price momentum
        for window in [5, 10, 20]:
            df[f'momentum_{window}d'] = df['close'] / df['close'].shift(window) - 1
        
        # Rate of Change
        for window in [5, 10, 20]:
            df[f'roc_{window}d'] = (df['close'] - df['close'].shift(window)) / df['close'].shift(window) * 100
        
        self.features_created.extend([
            'rsi_14', 'stoch_k', 'stoch_d',
            'macd', 'macd_signal', 'macd_histogram',
            'momentum_5d', 'momentum_10d', 'momentum_20d',
            'roc_5d', 'roc_10d', 'roc_20d'
        ])
        
        return df
    
    def create_volume_features(self, df):
        """Create volume-based features"""
        df = df.copy()
        
        # Volume moving averages
        for window in [5, 10, 20, 50]:
            df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_sma_{window}']
        
        # On Balance Volume
        obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        df['obv'] = obv.on_balance_volume()
        df['obv_sma_10'] = df['obv'].rolling(window=10).mean()
        df['obv_ratio'] = df['obv'] / df['obv_sma_10']
        
        # Volume Price Trend
        df['vpt'] = (df['volume'] * df['return_1d']).cumsum()
        
        # Price Volume
        df['price_volume'] = df['close'] * df['volume']
        df['price_volume_sma_10'] = df['price_volume'].rolling(window=10).mean()
        
        self.features_created.extend([
            'volume_sma_5', 'volume_sma_10', 'volume_sma_20', 'volume_sma_50',
            'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20', 'volume_ratio_50',
            'obv', 'obv_sma_10', 'obv_ratio', 'vpt', 'price_volume', 'price_volume_sma_10'
        ])
        
        return df
    
    def create_lagged_features(self, df, lags=[1, 2, 3, 5, 10]):
        """Create lagged features"""
        df = df.copy()
        
        # Key features to lag
        key_features = ['close', 'volume', 'return_1d', 'rsi_14', 'volatility_20d']
        
        for feature in key_features:
            if feature in df.columns:
                for lag in lags:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
                    self.features_created.append(f'{feature}_lag_{lag}')
        
        return df
    
    def create_target_variables(self, df):
        """Create target variables for prediction"""
        df = df.copy()
        
        # Next day price and return
        df['target_price_1d'] = df['close'].shift(-1)
        df['target_return_1d'] = df['return_1d'].shift(-1)
        
        # Next week price and return
        df['target_price_5d'] = df['close'].shift(-5)
        df['target_return_5d'] = df['return_5d'].shift(-5)
        
        # Price direction (classification target)
        df['target_direction_1d'] = np.where(df['target_return_1d'] > 0, 1, 0)
        df['target_direction_5d'] = np.where(df['target_return_5d'] > 0, 1, 0)
        
        return df

class SentimentFeatureEngineer:
    """Feature engineering for sentiment data"""
    
    def __init__(self):
        self.features_created = []
    
    def create_sentiment_features(self, df):
        """Create sentiment-based features"""
        df = df.copy()
        df = df.sort_values(['Ticker', 'Date'])
        
        # Rolling sentiment averages
        for window in [3, 7, 14, 30]:
            df[f'sentiment_sma_{window}'] = df.groupby('Ticker')['sentiment_score'].rolling(window=window).mean().reset_index(0, drop=True)
        
        # Sentiment momentum
        for window in [3, 7, 14]:
            df[f'sentiment_momentum_{window}d'] = df.groupby('Ticker')['sentiment_score'].pct_change(window).reset_index(0, drop=True)
        
        # Sentiment volatility
        for window in [7, 14, 30]:
            df[f'sentiment_volatility_{window}d'] = df.groupby('Ticker')['sentiment_score'].rolling(window=window).std().reset_index(0, drop=True)
        
        # News count features
        for window in [3, 7, 14]:
            df[f'news_count_sma_{window}'] = df.groupby('Ticker')['news_count'].rolling(window=window).mean().reset_index(0, drop=True)
        
        # Sentiment extremes
        df['sentiment_extreme'] = np.where(np.abs(df['sentiment_score']) > 0.5, 1, 0)
        df['sentiment_positive'] = np.where(df['sentiment_score'] > 0.1, 1, 0)
        df['sentiment_negative'] = np.where(df['sentiment_score'] < -0.1, 1, 0)
        
        # Lagged sentiment
        for lag in [1, 2, 3, 5, 7]:
            df[f'sentiment_lag_{lag}'] = df.groupby('Ticker')['sentiment_score'].shift(lag).reset_index(0, drop=True)
        
        self.features_created.extend([
            'sentiment_sma_3', 'sentiment_sma_7', 'sentiment_sma_14', 'sentiment_sma_30',
            'sentiment_momentum_3d', 'sentiment_momentum_7d', 'sentiment_momentum_14d',
            'sentiment_volatility_7d', 'sentiment_volatility_14d', 'sentiment_volatility_30d',
            'news_count_sma_3', 'news_count_sma_7', 'news_count_sma_14',
            'sentiment_extreme', 'sentiment_positive', 'sentiment_negative',
            'sentiment_lag_1', 'sentiment_lag_2', 'sentiment_lag_3', 'sentiment_lag_5', 'sentiment_lag_7'
        ])
        
        return df
    
    def forward_fill_sentiment(self, df):
        """Forward fill missing sentiment data per ticker"""
        df = df.copy()
        df = df.sort_values(['Ticker', 'Date'])
        
        # Forward fill sentiment scores within each ticker
        df['sentiment_score'] = df.groupby('Ticker')['sentiment_score'].fillna(method='ffill')
        df['news_count'] = df.groupby('Ticker')['news_count'].fillna(method='ffill')
        
        return df

class MacroFeatureEngineer:
    """Feature engineering for macroeconomic data"""
    
    def __init__(self):
        self.features_created = []
    
    def create_macro_features(self, df):
        """Create macro-based features"""
        df = df.copy()
        df = df.sort_values('Date')
        
        macro_cols = [col for col in df.columns if col != 'Date']
        
        # Rate of change for macro indicators
        for col in macro_cols:
            for window in [30, 90, 180]:  # Monthly, quarterly, semi-annual
                df[f'{col}_roc_{window}d'] = df[col].pct_change(window)
                self.features_created.append(f'{col}_roc_{window}d')
        
        # Moving averages for macro indicators
        for col in macro_cols:
            for window in [30, 90]:
                df[f'{col}_sma_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_ratio_{window}'] = df[col] / df[f'{col}_sma_{window}']
                self.features_created.extend([f'{col}_sma_{window}', f'{col}_ratio_{window}'])
        
        # Macro regime indicators
        df['unemployment_high'] = np.where(df['unemployment_rate'] > df['unemployment_rate'].rolling(252).quantile(0.75), 1, 0)
        df['inflation_high'] = np.where(df['inflation_rate'] > df['inflation_rate'].rolling(252).quantile(0.75), 1, 0)
        df['rates_rising'] = np.where(df['federal_funds_rate'] > df['federal_funds_rate'].shift(30), 1, 0)
        df['vix_high'] = np.where(df['vix'] > 25, 1, 0)  # VIX > 25 indicates high volatility
        
        self.features_created.extend(['unemployment_high', 'inflation_high', 'rates_rising', 'vix_high'])
        
        return df

class DataMerger:
    """Class to merge all datasets with proper handling"""
    
    def __init__(self):
        pass
    
    def merge_all_data(self, stock_data, sentiment_data, macro_data):
        """Merge stock, sentiment, and macro data"""
        print("Merging all datasets...")
        
        # Start with stock data as base
        merged_data = stock_data.copy()
        print(f"Starting with stock data: {merged_data.shape}")
        
        # Merge sentiment data
        sentiment_cols = [col for col in sentiment_data.columns if col not in ['Date', 'Ticker']]
        merged_data = pd.merge(
            merged_data,
            sentiment_data[['Date', 'Ticker'] + sentiment_cols],
            on=['Date', 'Ticker'],
            how='left'
        )
        print(f"After merging sentiment: {merged_data.shape}")
        
        # Merge macro data (macro data doesn't have Ticker, so merge only on Date)
        macro_cols = [col for col in macro_data.columns if col != 'Date']
        merged_data = pd.merge(
            merged_data,
            macro_data[['Date'] + macro_cols],
            on='Date',
            how='left'
        )
        print(f"After merging macro data: {merged_data.shape}")
        
        # Sort by Ticker and Date
        merged_data = merged_data.sort_values(['Ticker', 'Date'])
        
        return merged_data
    
    def clean_merged_data(self, df):
        """Clean merged data and handle missing values"""
        df = df.copy()
        
        print(f"Before cleaning: {df.shape}")
        print(f"Missing values:\n{df.isnull().sum().sum()}")
        
        # Forward fill macro data (since it's less frequent)
        macro_cols = ['GDP', 'unemployment_rate', 'inflation_rate', 'federal_funds_rate', 'consumer_sentiment', 'vix']
        for col in macro_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
        
        # Forward fill sentiment data per ticker
        sentiment_cols = [col for col in df.columns if 'sentiment' in col or 'news_count' in col]
        for col in sentiment_cols:
            if col in df.columns:
                df[col] = df.groupby('Ticker')[col].fillna(method='ffill')
        
        # Drop rows with remaining missing values in key columns
        key_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df.dropna(subset=key_cols)
        
        print(f"After cleaning: {df.shape}")
        print(f"Remaining missing values: {df.isnull().sum().sum()}")
        
        return df

def main():
    """Main function to run feature engineering pipeline"""
    
    print("Starting feature engineering pipeline...")
    
    # Load data
    print("Loading data...")
    stock_data = pd.read_csv('data/stock_prices.csv')
    sentiment_data = pd.read_csv('data/news_sentiment.csv')
    macro_data = pd.read_csv('data/macro_indicators.csv')
    
    # Convert dates
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'])
    macro_data['Date'] = pd.to_datetime(macro_data['Date'])
    
    # Initialize feature engineers
    stock_fe = StockFeatureEngineer()
    sentiment_fe = SentimentFeatureEngineer()
    macro_fe = MacroFeatureEngineer()
    merger = DataMerger()
    
    # Process sentiment data first (includes forward fill)
    print("Processing sentiment data...")
    sentiment_data = sentiment_fe.forward_fill_sentiment(sentiment_data)
    sentiment_data = sentiment_fe.create_sentiment_features(sentiment_data)
    
    # Process macro data
    print("Processing macro data...")
    macro_data = macro_fe.create_macro_features(macro_data)
    
    # Process stock data by ticker
    print("Processing stock data...")
    processed_stock_data = []
    
    for ticker in stock_data['Ticker'].unique():
        print(f"Processing {ticker}...")
        ticker_data = stock_data[stock_data['Ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date')
        
        # Create features
        ticker_data = stock_fe.create_price_features(ticker_data)
        ticker_data = stock_fe.create_moving_averages(ticker_data)
        ticker_data = stock_fe.create_volatility_features(ticker_data)
        ticker_data = stock_fe.create_momentum_features(ticker_data)
        ticker_data = stock_fe.create_volume_features(ticker_data)
        ticker_data = stock_fe.create_lagged_features(ticker_data)
        ticker_data = stock_fe.create_target_variables(ticker_data)
        
        processed_stock_data.append(ticker_data)
    
    # Combine all processed stock data
    processed_stock_data = pd.concat(processed_stock_data, ignore_index=True)
    
    # Merge all datasets
    print("Merging all datasets...")
    final_data = merger.merge_all_data(processed_stock_data, sentiment_data, macro_data)
    final_data = merger.clean_merged_data(final_data)
    
    # Save processed data
    print("Saving processed data...")
    final_data.to_csv('data/processed_features.csv', index=False)
    
    print(f"Feature engineering completed!")
    print(f"Final dataset shape: {final_data.shape}")
    print(f"Total features created: {len(final_data.columns)}")
    print(f"Features available: {sorted(final_data.columns.tolist())}")
    
    # Save feature summary
    feature_summary = {
        'total_features': len(final_data.columns),
        'stock_features': len(stock_fe.features_created),
        'sentiment_features': len(sentiment_fe.features_created),
        'macro_features': len(macro_fe.features_created),
        'feature_list': final_data.columns.tolist()
    }
    
    import json
    with open('data/feature_summary.json', 'w') as f:
        json.dump(feature_summary, f, indent=2, default=str)
    
    print("Feature summary saved to data/feature_summary.json")

if __name__ == "__main__":
    main()
