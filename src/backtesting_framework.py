#!/usr/bin/env python3
"""
Comprehensive Backtesting Framework for Stock Market AI Agent

This module implements a robust backtesting framework to validate trading strategies 
using historical stock data and predictions from the optimized AI models.

Features:
- Multiple trading strategies
- Comprehensive performance metrics
- Risk analysis and drawdown calculations
- Portfolio simulation
- Benchmark comparisons
- Detailed reporting

Author: AI Assistant
Date: July 2025
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate_signals(self, data: pd.DataFrame, predictions: np.ndarray) -> pd.Series:
        """Generate trading signals based on predictions"""
        raise NotImplementedError("Subclasses must implement generate_signals")

class MomentumStrategy(TradingStrategy):
    """Simple momentum strategy based on price predictions"""
    
    def __init__(self, threshold: float = 0.02):
        super().__init__("Momentum Strategy")
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame, predictions: np.ndarray) -> pd.Series:
        """
        Generate signals:
        1: Buy signal (predicted return > threshold)
        -1: Sell signal (predicted return < -threshold)
        0: Hold
        """
        current_prices = data['close'].values
        predicted_returns = (predictions - current_prices) / current_prices
        
        signals = np.zeros(len(predictions))
        signals[predicted_returns > self.threshold] = 1
        signals[predicted_returns < -self.threshold] = -1
        
        return pd.Series(signals, index=data.index)

class MeanReversionStrategy(TradingStrategy):
    """Mean reversion strategy"""
    
    def __init__(self, lookback_period: int = 20, threshold: float = 2.0):
        super().__init__("Mean Reversion Strategy")
        self.lookback_period = lookback_period
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame, predictions: np.ndarray) -> pd.Series:
        """Generate mean reversion signals"""
        prices = data['close']
        rolling_mean = prices.rolling(window=self.lookback_period).mean()
        rolling_std = prices.rolling(window=self.lookback_period).std()
        
        # Z-score calculation
        z_scores = (prices - rolling_mean) / rolling_std
        
        signals = np.zeros(len(data))
        # Buy when price is oversold and prediction is positive
        signals[(z_scores < -self.threshold) & (predictions > prices)] = 1
        # Sell when price is overbought and prediction is negative
        signals[(z_scores > self.threshold) & (predictions < prices)] = -1
        
        return pd.Series(signals, index=data.index)

class BacktestingEngine:
    """Comprehensive backtesting engine"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = {}
        
    def load_model_and_features(self, model_path: str, feature_names_path: str):
        """Load the trained model and feature names"""
        print(f"📊 Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(feature_names_path, 'rb') as f:
            self.feature_names = pickle.load(f)
        
        print(f"✅ Model loaded successfully with {len(self.feature_names)} features")
    
    def prepare_data(self, data_path: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load and prepare historical data for backtesting"""
        print(f"📈 Loading historical data from {data_path}")
        
        data = pd.read_csv(data_path)
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Filter date range if specified
        if start_date:
            data = data[data['Date'] >= start_date]
        if end_date:
            data = data[data['Date'] <= end_date]
        
        # Sort by ticker and date
        data = data.sort_values(['Ticker', 'Date'])
        
        # Clean data
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        print(f"✅ Data prepared: {len(data)} records from {data['Date'].min()} to {data['Date'].max()}")
        print(f"📋 Tickers: {data['Ticker'].nunique()} unique stocks")
        
        return data
    
    def generate_predictions(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the loaded model"""
        print("🔮 Generating price predictions...")
        
        # Prepare features for prediction
        # Use only features that exist in both the model and the data
        available_features = [f for f in self.feature_names if f in data.columns]
        missing_features = [f for f in self.feature_names if f not in data.columns]
        
        if missing_features:
            print(f"⚠️ Warning: {len(missing_features)} features missing from data")
            print(f"Missing: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
        
        print(f"✅ Using {len(available_features)} available features")
        
        # Create feature data with available features and fill missing with zeros
        feature_data = pd.DataFrame(index=data.index)
        for feature in self.feature_names:
            if feature in data.columns:
                feature_data[feature] = data[feature].fillna(0)
            else:
                feature_data[feature] = 0  # Fill missing features with zeros
        
        # Generate predictions
        predictions = self.model.predict(feature_data)
        
        print(f"✅ Generated {len(predictions)} predictions")
        return predictions
    
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    strategy: TradingStrategy,
                    ticker: str = None,
                    rebalance_frequency: str = 'daily') -> Dict:
        """Run backtest for a specific strategy"""
        
        print(f"\n🚀 Running backtest for {strategy.name}")
        print("=" * 60)
        
        if ticker:
            data = data[data['Ticker'] == ticker].copy()
            print(f"📊 Backtesting single ticker: {ticker}")
        
        if len(data) == 0:
            raise ValueError("No data available for backtesting")
        
        # Generate predictions
        predictions = self.generate_predictions(data)
        
        # Generate trading signals
        signals = strategy.generate_signals(data, predictions)
        
        # Initialize portfolio tracking
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},  # ticker -> shares
            'portfolio_value': [],
            'dates': [],
            'trades': [],
            'returns': []
        }
        
        # Simulate trading
        for i, (idx, row) in enumerate(data.iterrows()):
            date = row['Date']
            ticker_name = row['Ticker']
            price = row['close']
            signal = signals.iloc[i]
            
            # Calculate current portfolio value
            position_value = portfolio['positions'].get(ticker_name, 0) * price
            current_value = portfolio['cash'] + position_value
            
            # Execute trades based on signals
            if signal == 1:  # Buy signal
                # Use 10% of portfolio value for each trade
                trade_amount = current_value * 0.1
                shares_to_buy = int(trade_amount / price)
                cost = shares_to_buy * price * (1 + self.commission)
                
                if cost <= portfolio['cash']:
                    portfolio['cash'] -= cost
                    portfolio['positions'][ticker_name] = portfolio['positions'].get(ticker_name, 0) + shares_to_buy
                    
                    portfolio['trades'].append({
                        'date': date,
                        'ticker': ticker_name,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': price,
                        'cost': cost
                    })
            
            elif signal == -1:  # Sell signal
                shares_held = portfolio['positions'].get(ticker_name, 0)
                if shares_held > 0:
                    proceeds = shares_held * price * (1 - self.commission)
                    portfolio['cash'] += proceeds
                    portfolio['positions'][ticker_name] = 0
                    
                    portfolio['trades'].append({
                        'date': date,
                        'ticker': ticker_name,
                        'action': 'SELL',
                        'shares': shares_held,
                        'price': price,
                        'proceeds': proceeds
                    })
            
            # Record portfolio value
            total_position_value = sum([shares * price for ticker_name, shares in portfolio['positions'].items() 
                                      if ticker_name == row['Ticker']])
            total_value = portfolio['cash'] + total_position_value
            
            portfolio['portfolio_value'].append(total_value)
            portfolio['dates'].append(date)
            
            # Calculate daily returns
            if len(portfolio['portfolio_value']) > 1:
                daily_return = (total_value - portfolio['portfolio_value'][-2]) / portfolio['portfolio_value'][-2]
                portfolio['returns'].append(daily_return)
            else:
                portfolio['returns'].append(0.0)
        
        # Calculate performance metrics
        results = self.calculate_performance_metrics(portfolio, data)
        results['strategy'] = strategy.name
        results['trades'] = portfolio['trades']
        
        print(f"✅ Backtest completed!")
        print(f"📊 Total trades: {len(portfolio['trades'])}")
        print(f"💰 Final portfolio value: ${results['final_value']:,.2f}")
        print(f"📈 Total return: {results['total_return']:.2%}")
        
        return results
    
    def calculate_performance_metrics(self, portfolio: Dict, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        portfolio_values = np.array(portfolio['portfolio_value'])
        returns = np.array(portfolio['returns'])
        dates = portfolio['dates']
        
        # Basic metrics
        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Annualized metrics
        days = (pd.to_datetime(dates[-1]) - pd.to_datetime(dates[0])).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Risk metrics
        returns_std = np.std(returns) * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = (annualized_return - 0.02) / returns_std if returns_std > 0 else 0  # Assuming 2% risk-free rate
        
        # Drawdown calculation
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        trades = portfolio['trades']
        profitable_trades = 0
        total_trades = len([t for t in trades if t['action'] == 'SELL'])
        
        for i, trade in enumerate(trades):
            if trade['action'] == 'SELL' and i > 0:
                # Find corresponding buy trade
                buy_trades = [t for t in trades[:i] if t['action'] == 'BUY' and t['ticker'] == trade['ticker']]
                if buy_trades:
                    avg_buy_price = np.mean([t['price'] for t in buy_trades])
                    if trade['price'] > avg_buy_price:
                        profitable_trades += 1
        
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Benchmark comparison (simple buy and hold)
        if len(data) > 0:
            first_price = data['close'].iloc[0]
            last_price = data['close'].iloc[-1]
            benchmark_return = (last_price - first_price) / first_price
            excess_return = total_return - benchmark_return
        else:
            benchmark_return = 0
            excess_return = total_return
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': returns_std,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'benchmark_return': benchmark_return,
            'excess_return': excess_return,
            'portfolio_values': portfolio_values.tolist(),
            'dates': [str(d) for d in dates],
            'daily_returns': returns.tolist()
        }
    
    def compare_strategies(self, data: pd.DataFrame, strategies: List[TradingStrategy], ticker: str = None) -> pd.DataFrame:
        """Compare multiple strategies"""
        
        print(f"\n🏆 STRATEGY COMPARISON")
        print("=" * 80)
        
        results = []
        for strategy in strategies:
            try:
                result = self.run_backtest(data, strategy, ticker)
                results.append(result)
            except Exception as e:
                print(f"❌ Error running {strategy.name}: {e}")
                continue
        
        # Create comparison DataFrame
        comparison_data = []
        for result in results:
            comparison_data.append({
                'Strategy': result['strategy'],
                'Total Return': f"{result['total_return']:.2%}",
                'Annualized Return': f"{result['annualized_return']:.2%}",
                'Volatility': f"{result['volatility']:.2%}",
                'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{result['max_drawdown']:.2%}",
                'Win Rate': f"{result['win_rate']:.2%}",
                'Total Trades': result['total_trades'],
                'Final Value': f"${result['final_value']:,.2f}",
                'Excess Return': f"{result['excess_return']:.2%}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save results
        self.results = results
        
        return comparison_df
    
    def generate_report(self, output_path: str = "backtest_report.html"):
        """Generate comprehensive backtest report"""
        
        if not self.results:
            print("❌ No backtest results to report. Run backtests first.")
            return
        
        print(f"📄 Generating comprehensive backtest report...")
        
        # Create detailed report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'commission': self.commission,
            'results': self.results
        }
        
        # Save to JSON
        json_path = output_path.replace('.html', '.json')
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"✅ Backtest report saved to {json_path}")
        
        return report_data

def main():
    """Main function to run backtesting"""
    
    print("🚀 STOCK MARKET AI AGENT - BACKTESTING FRAMEWORK")
    print("=" * 80)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # Initialize backtesting engine
        engine = BacktestingEngine(initial_capital=100000, commission=0.001)
        
        # Load model
        engine.load_model_and_features(
            'models/random_forest_model.pkl',
            'models/feature_names.pkl'
        )
        
        # Prepare data
        data = engine.prepare_data('data/processed_features.csv')
        
        # Define strategies
        strategies = [
            MomentumStrategy(threshold=0.02),
            MomentumStrategy(threshold=0.01),
            MeanReversionStrategy(lookback_period=20, threshold=2.0)
        ]
        
        # Run strategy comparison
        comparison = engine.compare_strategies(data, strategies, ticker='AAPL')
        
        print(f"\n🏆 STRATEGY COMPARISON RESULTS")
        print("=" * 80)
        print(comparison.to_string(index=False))
        
        # Generate report
        engine.generate_report('outputs/backtest_results.json')
        
        print(f"\n✅ BACKTESTING COMPLETE!")
        print("=" * 80)
        print("📊 Check outputs/backtest_results.json for detailed results")
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
