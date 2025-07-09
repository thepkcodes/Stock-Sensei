import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM models will be disabled.")

import warnings
warnings.filterwarnings('ignore')

class AdvancedModelBuilder:
    """Advanced class for building and evaluating stock prediction models"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and clean the processed feature data"""
        print("Loading data...")
        self.data = pd.read_csv(self.data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Replace inf values with NaN and drop problematic rows
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        initial_shape = self.data.shape
        self.data.dropna(inplace=True)
        final_shape = self.data.shape
        
        print(f"Data loaded: {initial_shape} -> {final_shape} (removed {initial_shape[0] - final_shape[0]} rows with NaN/inf)")
        
    def prepare_data(self, target_col='target_price_1d', test_size=0.2):
        """Prepare data for model training with proper time series handling"""
        print(f"Preparing data with target: {target_col}...")
        
        # Sort by date to maintain temporal order
        data = self.data.copy()
        data = data.sort_values(['Ticker', 'Date'])
        data = data.dropna(subset=[target_col])
        
        # Drop target columns we're not using
        drop_cols = ['Date', 'Ticker']
        target_cols = [col for col in data.columns if col.startswith('target_')]
        drop_cols.extend([col for col in target_cols if col != target_col])
        
        X = data.drop(columns=drop_cols)
        y = data[target_col]
        
        # Time series split to prevent data leakage
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        
        print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, **kwargs):
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        model = RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 15),
            min_samples_split=kwargs.get('min_samples_split', 5),
            min_samples_leaf=kwargs.get('min_samples_leaf', 2),
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        print("Random Forest training completed")
        
        return model
    
    def train_xgboost(self, X_train, y_train, **kwargs):
        """Train XGBoost model"""
        print("Training XGBoost model...")
        
        model = xgb.XGBRegressor(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.1),
            subsample=kwargs.get('subsample', 0.8),
            colsample_bytree=kwargs.get('colsample_bytree', 0.8),
            random_state=42,
            n_jobs=-1,
            eval_metric='rmse'
        )
        
        model.fit(X_train, y_train)
        print("XGBoost training completed")
        
        return model
    
    def prepare_lstm_data(self, X, y, sequence_length=60):
        """Prepare data for LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
            
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(X_scaled)):
            sequences.append(X_scaled[i-sequence_length:i])
            targets.append(y.iloc[i])
        
        return np.array(sequences), np.array(targets)
    
    def train_lstm(self, X_train, y_train, **kwargs):
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Skipping LSTM training.")
            return None
            
        print("Training LSTM model...")
        
        sequence_length = kwargs.get('sequence_length', 60)
        
        # Prepare sequences
        X_seq, y_seq = self.prepare_lstm_data(X_train, y_train, sequence_length)
        
        if len(X_seq) == 0:
            print("Not enough data for LSTM sequences. Skipping LSTM training.")
            return None
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        model.fit(X_seq, y_seq, 
                 batch_size=kwargs.get('batch_size', 32),
                 epochs=kwargs.get('epochs', 50),
                 verbose=0)
        
        print("LSTM training completed")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Evaluate model performance"""
        print(f"Evaluating {model_name}...")
        
        if model_name == "LSTM" and TENSORFLOW_AVAILABLE:
            # For LSTM, we need to prepare sequences
            X_seq, y_seq = self.prepare_lstm_data(X_test, y_test)
            if len(X_seq) == 0:
                print("Not enough test data for LSTM evaluation")
                return None, None, None
            predictions = model.predict(X_seq, verbose=0).flatten()
            y_actual = y_seq
        else:
            predictions = model.predict(X_test)
            y_actual = y_test
        
        mse = mean_squared_error(y_actual, predictions)
        mae = mean_absolute_error(y_actual, predictions)
        r2 = r2_score(y_actual, predictions)
        rmse = np.sqrt(mse)
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        return rmse, mae, r2, predictions, y_actual
    
    def plot_predictions(self, models_results, save_path=None):
        """Plot predictions vs actual values for all models"""
        n_models = len(models_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (name, (rmse, mae, r2, predictions, y_actual)) in enumerate(models_results.items()):
            if predictions is None:
                continue
                
            axes[i].scatter(y_actual, predictions, alpha=0.6, s=10)
            
            # Perfect prediction line
            min_val = min(y_actual.min(), predictions.min())
            max_val = max(y_actual.max(), predictions.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            axes[i].set_xlabel('Actual Prices')
            axes[i].set_ylabel('Predicted Prices')
            axes[i].set_title(f'{name}\\nRMSE: {rmse:.4f}, R²: {r2:.4f}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_feature_importance(self, model, feature_names, model_name="Model", top_n=20):
        """Get and plot feature importance"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\\n🔍 Top {top_n} Most Important Features ({model_name}):")
            print(importance_df.head(top_n))
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(top_n)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Features - {model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            return importance_df
        else:
            print(f"Feature importance not available for {model_name}")
            return None
    
    def run_comprehensive_pipeline(self, target_col='target_price_1d'):
        """Run comprehensive modeling pipeline"""
        print("🚀 Starting Comprehensive Stock Prediction Pipeline")
        print("=" * 60)
        
        # Load and prepare data
        self.load_data()
        X_train, X_test, y_train, y_test = self.prepare_data(target_col=target_col)
        
        models = {}
        results = {}
        
        # Train Random Forest
        print("\\n1️⃣ Random Forest")
        print("-" * 20)
        rf_model = self.train_random_forest(X_train, y_train)
        rf_results = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
        models['Random Forest'] = rf_model
        results['Random Forest'] = rf_results
        
        # Train XGBoost
        print("\\n2️⃣ XGBoost")
        print("-" * 20)
        xgb_model = self.train_xgboost(X_train, y_train)
        xgb_results = self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        models['XGBoost'] = xgb_model
        results['XGBoost'] = xgb_results
        
        # Train LSTM (if available)
        if TENSORFLOW_AVAILABLE:
            print("\\n3️⃣ LSTM")
            print("-" * 20)
            lstm_model = self.train_lstm(X_train, y_train)
            if lstm_model is not None:
                lstm_results = self.evaluate_model(lstm_model, X_test, y_test, "LSTM")
                models['LSTM'] = lstm_model
                results['LSTM'] = lstm_results
        
        # Compare results
        print("\\n📊 Model Comparison")
        print("=" * 60)
        performance_df = pd.DataFrame([
            {'Model': name, 'RMSE': res[0], 'MAE': res[1], 'R²': res[2]}
            for name, res in results.items() if res[0] is not None
        ])
        
        print(performance_df.round(4))
        
        # Find best model
        best_model_name = performance_df.loc[performance_df['R²'].idxmax(), 'Model']
        print(f"\\n🏆 Best Model: {best_model_name}")
        
        # Plot predictions
        print("\\n📈 Plotting Predictions...")
        plot_results = {name: res for name, res in results.items() if res[0] is not None}
        self.plot_predictions(plot_results, save_path='outputs/model_predictions.png')
        
        # Feature importance for tree models
        for model_name, model in models.items():
            if model_name in ['Random Forest', 'XGBoost']:
                self.get_feature_importance(model, X_train.columns, model_name)
        
        # Save results
        performance_df.to_csv('outputs/model_performance.csv', index=False)
        print("\\n💾 Results saved to outputs/")
        
        return models, results, performance_df

def main():
    """Main function"""
    builder = AdvancedModelBuilder(data_path='data/processed_features.csv')
    models, results, performance_df = builder.run_comprehensive_pipeline()

if __name__ == "__main__":
    main()
