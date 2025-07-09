import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from model_builder import ModelBuilder
import warnings
warnings.filterwarnings('ignore')

def train_and_save_models():
    """Train models and save them for deployment"""
    print("🔧 Training and Saving Final Models...")
    
    # Initialize model builder
    builder = ModelBuilder(data_path='data/processed_features.csv')
    builder.load_data()
    
    # Clean data
    builder.data.replace([np.inf, -np.inf], np.nan, inplace=True)
    builder.data.dropna(inplace=True)
    
    # Prepare data
    X_train, X_test, y_train, y_test = builder.prepare_data(target_col='target_price_1d')
    
    print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    
    # Train Random Forest (best model)
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Train XGBoost (backup model)
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='rmse'
    )
    xgb_model.fit(X_train, y_train)
    
    # Evaluate models
    rf_score = rf_model.score(X_test, y_test)
    xgb_score = xgb_model.score(X_test, y_test)
    
    print(f"Random Forest R²: {rf_score:.4f}")
    print(f"XGBoost R²: {xgb_score:.4f}")
    
    # Save models
    print("Saving models...")
    
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    
    # Save feature names
    feature_names = X_train.columns.tolist()
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save model metadata
    metadata = {
        'random_forest_r2': rf_score,
        'xgboost_r2': xgb_score,
        'n_features': len(feature_names),
        'n_training_samples': len(X_train),
        'target_column': 'target_price_1d',
        'feature_names': feature_names
    }
    
    with open('models/model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("✅ Models saved successfully!")
    print(f"📁 Random Forest: models/random_forest_model.pkl")
    print(f"📁 XGBoost: models/xgboost_model.pkl")
    print(f"📁 Features: models/feature_names.pkl")
    print(f"📁 Metadata: models/model_metadata.pkl")
    
    return rf_model, xgb_model, feature_names

if __name__ == "__main__":
    train_and_save_models()
