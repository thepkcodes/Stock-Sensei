import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class ModelBuilder:
    """Class to build and evaluate machine learning models"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None

    def load_data(self):
        """Load the processed feature data"""
        print("Loading data...")
        self.data = pd.read_csv(self.data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        print(f"Data loaded: {self.data.shape}")

    def prepare_data(self, target_col='target_price_1d'):
        """Prepare data for model training"""
        print(f"Preparing data with target: {target_col}...")

        # Drop rows with missing target or key columns
        data = self.data.copy()
        data = data.dropna(subset=[target_col])

        # Features and target
        X = data.drop(columns=['Date', 'Ticker', target_col, 'target_price_5d', 'target_return_1d', 'target_return_5d'])
        y = data[target_col]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train, model_type='random_forest'):
        """Train specified model and return the trained model"""
        print(f"Training {model_type} model...")

        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'xgboost':
            model = xgb.XGBRegressor(n_estimators=100, use_label_encoder=False, eval_metric='rmse', random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.fit(X_train, y_train)
        print(f"Model training completed: {model_type}")

        return model

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model's performance on test data"""
        print("Evaluating model...")

        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R2: {r2:.2f}")

        return mse, mae, r2

    def run_pipeline(self, target_col='target_price_1d'):
        """Run the full modeling pipeline for Random Forest and XGBoost"""
        self.load_data()
        
        # Replace inf values with NaN and drop these rows
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.dropna(inplace=True)
        
        X_train, X_test, y_train, y_test = self.prepare_data(target_col=target_col)

        # Random Forest
        rf_model = self.train_model(X_train, y_train, model_type='random_forest')
        print("\nRandom Forest Results:")
        self.evaluate_model(rf_model, X_test, y_test)

        # XGBoost
        xgb_model = self.train_model(X_train, y_train, model_type='xgboost')
        print("\nXGBoost Results:")
        self.evaluate_model(xgb_model, X_test, y_test)

if __name__ == "__main__":
    builder = ModelBuilder(data_path='data/processed_features.csv')
    builder.run_pipeline()
