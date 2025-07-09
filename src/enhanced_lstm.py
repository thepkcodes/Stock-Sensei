import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, GRU
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Please install it to use LSTM models.")

class EnhancedLSTMPredictor:
    """Enhanced LSTM model for stock price prediction with proper time series architecture"""
    
    def __init__(self, data_path, sequence_length=60):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.feature_columns = None
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
    
    def load_and_prepare_data(self, target_col='target_price_1d'):
        """Load and prepare data for LSTM training"""
        print("Loading data for LSTM...")
        
        # Load data
        data = pd.read_csv(self.data_path)
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Clean data
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        
        # Sort by ticker and date
        data = data.sort_values(['Ticker', 'Date'])
        data = data.dropna(subset=[target_col])
        
        # Prepare features and target
        drop_cols = ['Date', 'Ticker']
        target_cols = [col for col in data.columns if col.startswith('target_')]
        drop_cols.extend([col for col in target_cols if col != target_col])
        
        X = data.drop(columns=drop_cols)
        y = data[target_col]
        
        self.feature_columns = X.columns.tolist()
        
        print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, data[['Ticker', 'Date']]
    
    def create_sequences_by_ticker(self, X, y, metadata):
        """Create sequences for each ticker separately to maintain temporal order"""
        print(f"Creating sequences with length {self.sequence_length}...")
        
        all_sequences_X = []
        all_sequences_y = []
        
        # Group by ticker to create sequences within each stock
        for ticker in metadata['Ticker'].unique():
            ticker_mask = metadata['Ticker'] == ticker
            # Handle both pandas and numpy arrays
            if hasattr(X, 'iloc'):
                ticker_X = X[ticker_mask].values
            else:
                ticker_X = X[ticker_mask]
            if hasattr(y, 'iloc'):
                ticker_y = y[ticker_mask].values
            else:
                ticker_y = y[ticker_mask]
            
            # Create sequences for this ticker
            if len(ticker_X) > self.sequence_length:
                for i in range(self.sequence_length, len(ticker_X)):
                    all_sequences_X.append(ticker_X[i-self.sequence_length:i])
                    all_sequences_y.append(ticker_y[i])
        
        if len(all_sequences_X) == 0:
            raise ValueError(f"Not enough data to create sequences of length {self.sequence_length}")
        
        sequences_X = np.array(all_sequences_X)
        sequences_y = np.array(all_sequences_y)
        
        print(f"Created {len(sequences_X)} sequences")
        print(f"Sequence shape: {sequences_X.shape}")
        
        return sequences_X, sequences_y
    
    def prepare_lstm_data(self, X, y, metadata, test_size=0.2):
        """Prepare data specifically for LSTM training"""
        print("Preparing LSTM training data...")
        
        # Scale features
        X_scaled = self.scaler_X.fit_transform(X)
        # Handle both pandas Series and numpy arrays
        if hasattr(y, 'values'):
            y_array = y.values
        else:
            y_array = y
        y_scaled = self.scaler_y.fit_transform(y_array.reshape(-1, 1)).ravel()
        
        # Create sequences
        sequences_X, sequences_y = self.create_sequences_by_ticker(X_scaled, y_scaled, metadata)
        
        # Split into train/test (time series split)
        split_idx = int(len(sequences_X) * (1 - test_size))
        
        X_train = sequences_X[:split_idx]
        X_test = sequences_X[split_idx:]
        y_train = sequences_y[:split_idx]
        y_test = sequences_y[split_idx:]
        
        print(f"Training sequences: {X_train.shape}")
        print(f"Testing sequences: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def build_enhanced_lstm_model(self, input_shape):
        """Build an enhanced LSTM model with multiple architectures"""
        print("Building enhanced LSTM model...")
        
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            
            # Second LSTM layer with return sequences
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            
            # Third LSTM layer without return sequences
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])
        
        # Compile with adaptive learning rate
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print("Model architecture:")
        model.summary()
        
        return model
    
    def build_bidirectional_lstm(self, input_shape):
        """Build a bidirectional LSTM model"""
        print("Building bidirectional LSTM model...")
        
        model = Sequential([
            # Bidirectional LSTM layers
            Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            
            Bidirectional(LSTM(32, return_sequences=False)),
            Dropout(0.2),
            
            # Dense layers
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def build_gru_model(self, input_shape):
        """Build a GRU model as an alternative to LSTM"""
        print("Building GRU model...")
        
        model = Sequential([
            # GRU layers
            GRU(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            
            GRU(64, return_sequences=True),
            Dropout(0.2),
            
            GRU(32, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test, model_type='enhanced_lstm', 
                   epochs=100, batch_size=32, validation_split=0.2):
        """Train the LSTM model with proper callbacks"""
        print(f"Training {model_type} model...")
        
        # Build model based on type
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        if model_type == 'enhanced_lstm':
            self.model = self.build_enhanced_lstm_model(input_shape)
        elif model_type == 'bidirectional':
            self.model = self.build_bidirectional_lstm(input_shape)
        elif model_type == 'gru':
            self.model = self.build_gru_model(input_shape)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained LSTM model"""
        print("Evaluating LSTM model...")
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        
        # Inverse transform predictions and actual values
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_actual = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        
        print(f"LSTM Model Performance:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        return rmse, mae, r2, y_pred, y_actual
    
    def save_model(self, model_path='models/enhanced_lstm_model.h5', 
                   scaler_path='models/lstm_scalers.pkl'):
        """Save the trained LSTM model and scalers"""
        print("Saving LSTM model...")
        
        # Save the model
        self.model.save(model_path)
        
        # Save scalers and metadata
        model_data = {
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length
        }
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
        print(f"Scalers and metadata saved to {scaler_path}")
    
    def run_lstm_pipeline(self, target_col='target_price_1d', model_type='enhanced_lstm'):
        """Run the complete LSTM training pipeline"""
        print("🧠 Starting Enhanced LSTM Training Pipeline")
        print("=" * 60)
        
        # Load and prepare data
        X, y, metadata = self.load_and_prepare_data(target_col)
        
        # Prepare LSTM-specific data
        X_train, X_test, y_train, y_test = self.prepare_lstm_data(X, y, metadata)
        
        # Train model
        history = self.train_model(X_train, y_train, X_test, y_test, model_type=model_type)
        
        # Evaluate model
        rmse, mae, r2, y_pred, y_actual = self.evaluate_model(X_test, y_test)
        
        # Save model
        self.save_model()
        
        results = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred,
            'actual': y_actual,
            'history': history.history
        }
        
        print("\\n✅ LSTM Training Complete!")
        print(f"🎯 Final R² Score: {r2:.4f}")
        
        return self.model, results

def compare_lstm_models(data_path='data/processed_features.csv'):
    """Compare different LSTM architectures"""
    print("🔬 Comparing LSTM Model Architectures")
    print("=" * 60)
    
    model_types = ['enhanced_lstm', 'bidirectional', 'gru']
    results = {}
    
    for model_type in model_types:
        print(f"\\nTraining {model_type}...")
        print("-" * 40)
        
        try:
            predictor = EnhancedLSTMPredictor(data_path, sequence_length=60)
            model, result = predictor.run_lstm_pipeline(model_type=model_type)
            results[model_type] = result
            
            # Save each model with specific name
            model_path = f'models/{model_type}_model.h5'
            scaler_path = f'models/{model_type}_scalers.pkl'
            predictor.save_model(model_path, scaler_path)
            
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            results[model_type] = None
    
    # Compare results
    print("\\n🏆 LSTM Model Comparison:")
    print("=" * 60)
    
    comparison_data = []
    for model_type, result in results.items():
        if result is not None:
            comparison_data.append({
                'Model': model_type,
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                'R²': result['r2']
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.round(4))
        
        # Save comparison
        comparison_df.to_csv('models/lstm_model_comparison.csv', index=False)
        
        # Find best model
        best_model = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
        print(f"\\n🥇 Best LSTM Model: {best_model}")
    
    return results

def main():
    """Main function"""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Please install it to run LSTM models.")
        return None
    
    # Run single enhanced LSTM
    predictor = EnhancedLSTMPredictor('data/processed_features.csv')
    model, results = predictor.run_lstm_pipeline()
    
    # Compare different architectures
    comparison_results = compare_lstm_models()
    
    return model, results, comparison_results

if __name__ == "__main__":
    main()
