#!/usr/bin/env python3
"""
Comprehensive Live Market Predictor for All Stocks

This script gets live predictions for all 49 stocks in our training data
and saves the results with proper dates for analysis.

Author: AI Assistant
Date: July 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os
from live_market_predictor import LiveMarketPredictor
import warnings

warnings.filterwarnings('ignore')

class ComprehensivePredictor:
    """Comprehensive predictor for all stocks"""
    
    def __init__(self):
        self.predictor = LiveMarketPredictor(
            model_path='models/random_forest_model.pkl',
            feature_names_path='models/feature_names.pkl'
        )
        
        # Load the list of all tickers from training data
        self.load_training_tickers()
    
    def load_training_tickers(self):
        """Load the list of tickers from training data"""
        try:
            data = pd.read_csv('data/processed_features.csv')
            self.all_tickers = sorted(data['Ticker'].unique())
            print(f"✅ Loaded {len(self.all_tickers)} tickers from training data")
            return True
        except Exception as e:
            print(f"❌ Error loading training tickers: {e}")
            self.all_tickers = []
            return False
    
    def predict_all_stocks(self, days_ahead: int = 1, save_results: bool = True):
        """Get predictions for all stocks"""
        
        print(f"\n🚀 COMPREHENSIVE LIVE MARKET PREDICTIONS")
        print("=" * 80)
        print(f"📊 Predicting {len(self.all_tickers)} stocks for {days_ahead} days ahead")
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        all_results = {}
        successful_predictions = 0
        failed_predictions = 0
        
        for i, ticker in enumerate(self.all_tickers, 1):
            try:
                print(f"\n[{i:2d}/{len(self.all_tickers):2d}] Processing {ticker}...")
                
                # Get prediction
                result = self.predictor.predict_future_price(ticker, days_ahead)
                
                if 'error' not in result:
                    all_results[ticker] = result
                    successful_predictions += 1
                    
                    # Print summary
                    direction = "📈" if result['price_change_pct'] > 0 else "📉"
                    print(f"    {direction} {ticker}: ${result['current_price']:.2f} → ${result['predicted_price']:.2f} ({result['price_change_pct']:+.2f}%)")
                else:
                    all_results[ticker] = result
                    failed_predictions += 1
                    print(f"    ❌ {ticker}: {result['error']}")
                    
            except Exception as e:
                all_results[ticker] = {"error": str(e)}
                failed_predictions += 1
                print(f"    ❌ {ticker}: {e}")
        
        # Generate comprehensive summary
        summary = self.generate_summary(all_results, days_ahead)
        
        # Save results if requested
        if save_results:
            self.save_results(all_results, summary, days_ahead)
        
        # Display final summary
        self.display_summary(summary, successful_predictions, failed_predictions)
        
        return all_results, summary
    
    def generate_summary(self, results: dict, days_ahead: int) -> dict:
        """Generate comprehensive summary of predictions"""
        
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            return {"error": "No successful predictions"}
        
        # Calculate statistics
        changes = [r['price_change_pct'] for r in successful_results.values()]
        confidences = [r['confidence'] for r in successful_results.values()]
        current_prices = [r['current_price'] for r in successful_results.values()]
        predicted_prices = [r['predicted_price'] for r in successful_results.values()]
        
        # Market sentiment analysis
        bullish_stocks = [k for k, v in successful_results.items() if v['price_change_pct'] > 0]
        bearish_stocks = [k for k, v in successful_results.items() if v['price_change_pct'] < 0]
        
        # Top gainers and losers
        sorted_by_change = sorted(successful_results.items(), key=lambda x: x[1]['price_change_pct'], reverse=True)
        top_gainers = sorted_by_change[:10]
        top_losers = sorted_by_change[-10:]
        
        # High confidence predictions
        high_confidence = [(k, v) for k, v in successful_results.items() if v['confidence'] > 85]
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'days_ahead': days_ahead,
            'total_stocks': len(results),
            'successful_predictions': len(successful_results),
            'failed_predictions': len(results) - len(successful_results),
            'statistics': {
                'avg_change': np.mean(changes),
                'median_change': np.median(changes),
                'std_change': np.std(changes),
                'min_change': np.min(changes),
                'max_change': np.max(changes),
                'avg_confidence': np.mean(confidences),
                'avg_current_price': np.mean(current_prices),
                'avg_predicted_price': np.mean(predicted_prices)
            },
            'market_sentiment': {
                'bullish_count': len(bullish_stocks),
                'bearish_count': len(bearish_stocks),
                'bullish_percentage': (len(bullish_stocks) / len(successful_results)) * 100,
                'overall_sentiment': 'BULLISH' if np.mean(changes) > 0 else 'BEARISH',
                'bullish_stocks': bullish_stocks[:10],  # Top 10
                'bearish_stocks': bearish_stocks[:10]   # Top 10
            },
            'top_gainers': [{'ticker': k, 'change': v['price_change_pct'], 'confidence': v['confidence']} 
                           for k, v in top_gainers],
            'top_losers': [{'ticker': k, 'change': v['price_change_pct'], 'confidence': v['confidence']} 
                          for k, v in top_losers],
            'high_confidence_predictions': [{'ticker': k, 'change': v['price_change_pct'], 'confidence': v['confidence']} 
                                          for k, v in high_confidence]
        }
        
        return summary
    
    def save_results(self, results: dict, summary: dict, days_ahead: int):
        """Save results to files"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs/live_predictions', exist_ok=True)
        
        # Save detailed results
        detailed_file = f'outputs/live_predictions/detailed_predictions_{timestamp}.json'
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = f'outputs/live_predictions/summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create CSV for easy analysis
        csv_data = []
        for ticker, result in results.items():
            if 'error' not in result:
                csv_data.append({
                    'Ticker': ticker,
                    'Current_Price': result['current_price'],
                    'Predicted_Price': result['predicted_price'],
                    'Price_Change': result['price_change'],
                    'Price_Change_Pct': result['price_change_pct'],
                    'Confidence': result['confidence'],
                    'Current_Date': result['current_date'],
                    'Prediction_Date': result['prediction_date'],
                    'Days_Ahead': result['days_ahead'],
                    'Volatility': result['recent_volatility']
                })
            else:
                csv_data.append({
                    'Ticker': ticker,
                    'Current_Price': 'ERROR',
                    'Predicted_Price': 'ERROR',
                    'Price_Change': 'ERROR',
                    'Price_Change_Pct': 'ERROR',
                    'Confidence': 'ERROR',
                    'Current_Date': 'ERROR',
                    'Prediction_Date': 'ERROR',
                    'Days_Ahead': days_ahead,
                    'Volatility': 'ERROR'
                })
        
        csv_file = f'outputs/live_predictions/predictions_{timestamp}.csv'
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   📄 Detailed: {detailed_file}")
        print(f"   📊 Summary: {summary_file}")
        print(f"   📈 CSV: {csv_file}")
    
    def display_summary(self, summary: dict, successful: int, failed: int):
        """Display comprehensive summary"""
        
        if 'error' in summary:
            print(f"\n❌ Summary generation failed: {summary['error']}")
            return
        
        print(f"\n" + "🏆" * 80)
        print("COMPREHENSIVE MARKET PREDICTION SUMMARY")
        print("🏆" * 80)
        
        # Overall statistics
        stats = summary['statistics']
        sentiment = summary['market_sentiment']
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   ✅ Successful Predictions: {successful}/{successful + failed} ({(successful/(successful+failed)*100):.1f}%)")
        print(f"   📈 Average Expected Change: {stats['avg_change']:+.2f}%")
        print(f"   🎯 Average Confidence: {stats['avg_confidence']:.1f}%")
        print(f"   📊 Change Range: {stats['min_change']:+.2f}% to {stats['max_change']:+.2f}%")
        
        print(f"\n🌍 MARKET SENTIMENT: {sentiment['overall_sentiment']}")
        print(f"   📈 Bullish Stocks: {sentiment['bullish_count']} ({sentiment['bullish_percentage']:.1f}%)")
        print(f"   📉 Bearish Stocks: {sentiment['bearish_count']} ({100-sentiment['bullish_percentage']:.1f}%)")
        
        # Top gainers
        print(f"\n🚀 TOP 10 EXPECTED GAINERS:")
        for i, gainer in enumerate(summary['top_gainers'], 1):
            print(f"   {i:2d}. {gainer['ticker']:5s}: +{gainer['change']:.2f}% (Confidence: {gainer['confidence']:.1f}%)")
        
        # Top losers
        print(f"\n📉 TOP 10 EXPECTED DECLINERS:")
        for i, loser in enumerate(summary['top_losers'], 1):
            print(f"   {i:2d}. {loser['ticker']:5s}: {loser['change']:+.2f}% (Confidence: {loser['confidence']:.1f}%)")
        
        # High confidence predictions
        if summary['high_confidence_predictions']:
            print(f"\n🎯 HIGH CONFIDENCE PREDICTIONS (>85%):")
            for pred in summary['high_confidence_predictions'][:15]:  # Top 15
                direction = "📈" if pred['change'] > 0 else "📉"
                print(f"   {direction} {pred['ticker']:5s}: {pred['change']:+.2f}% (Confidence: {pred['confidence']:.1f}%)")
        
        print(f"\n" + "✅" * 80)
        print("PREDICTION ANALYSIS COMPLETE!")
        print("✅" * 80)

def main():
    """Main function"""
    
    print("🚀 COMPREHENSIVE STOCK MARKET PREDICTOR")
    print("=" * 60)
    
    # Initialize predictor
    predictor = ComprehensivePredictor()
    
    if not predictor.all_tickers:
        print("❌ Could not load stock tickers. Exiting.")
        return
    
    # Get predictions for 1 day ahead
    print(f"\nGetting live predictions for all {len(predictor.all_tickers)} stocks...")
    results, summary = predictor.predict_all_stocks(days_ahead=1, save_results=True)
    
    # Also get 5-day ahead predictions for comparison
    print(f"\n\n🔮 Getting 5-day ahead predictions...")
    results_5d, summary_5d = predictor.predict_all_stocks(days_ahead=5, save_results=True)
    
    print(f"\n🎉 All predictions completed!")
    print(f"📁 Check outputs/live_predictions/ for detailed results")

if __name__ == "__main__":
    main()
