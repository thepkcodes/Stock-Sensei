import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockPredictionService:
    """Production-ready service for stock price predictions"""
    
    def __init__(self, model_path='models/random_forest_model.pkl', 
                 feature_names_path='models/feature_names.pkl',
                 metadata_path='models/model_metadata.pkl'):
        self.model = None
        self.feature_names = None
        self.metadata = None
        self.model_path = model_path
        self.feature_names_path = feature_names_path
        self.metadata_path = metadata_path
        
    def load_model(self):
        """Load the trained model and metadata"""
        print("Loading trained model...")
        
        try:
            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Model loaded from {self.model_path}")
            
            # Load feature names
            with open(self.feature_names_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            print(f"✅ Feature names loaded ({len(self.feature_names)} features)")
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"✅ Metadata loaded")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def validate_input(self, input_data):
        """Validate input data for prediction"""
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        
        # Check if all required features are present
        missing_features = set(self.feature_names) - set(input_data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Check for missing values
        if input_data[self.feature_names].isnull().any().any():
            print("⚠️ Warning: Input data contains missing values. Filling with zeros.")
            input_data[self.feature_names] = input_data[self.feature_names].fillna(0)
        
        # Check for infinite values
        if np.isinf(input_data[self.feature_names]).any().any():
            print("⚠️ Warning: Input data contains infinite values. Replacing with zeros.")
            input_data[self.feature_names] = input_data[self.feature_names].replace([np.inf, -np.inf], 0)
        
        return input_data[self.feature_names]
    
    def predict(self, input_data, return_confidence=False):
        """Make stock price predictions"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Validate and prepare input
        validated_data = self.validate_input(input_data)
        
        # Make prediction
        prediction = self.model.predict(validated_data)
        
        # Calculate confidence if requested (for tree-based models)
        confidence = None
        if return_confidence and hasattr(self.model, 'predict_proba'):
            # For regression, we can use prediction variance from trees
            if hasattr(self.model, 'estimators_'):
                tree_predictions = np.array([tree.predict(validated_data) for tree in self.model.estimators_])
                confidence = np.std(tree_predictions, axis=0)
        
        return prediction, confidence
    
    def predict_single_stock(self, ticker_data, return_details=False):
        """Predict price for a single stock with detailed output"""
        prediction, confidence = self.predict(ticker_data, return_confidence=True)
        
        result = {
            'predicted_price': float(prediction[0]) if len(prediction) == 1 else prediction.tolist(),
            'timestamp': datetime.now().isoformat(),
            'model_version': self.metadata.get('model_version', '1.0'),
            'confidence_score': float(confidence[0]) if confidence is not None and len(confidence) == 1 else None
        }
        
        if return_details:
            result.update({
                'model_type': type(self.model).__name__,
                'features_used': len(self.feature_names),
                'training_r2': self.metadata.get('random_forest_r2', 'N/A')
            })
        
        return result
    
    def batch_predict(self, input_data, ticker_column='Ticker'):
        """Make predictions for multiple stocks"""
        results = {}
        
        if ticker_column in input_data.columns:
            # Group by ticker and predict for each
            for ticker in input_data[ticker_column].unique():
                ticker_data = input_data[input_data[ticker_column] == ticker]
                ticker_features = ticker_data.drop(columns=[ticker_column])
                
                prediction, confidence = self.predict(ticker_features, return_confidence=True)
                
                results[ticker] = {
                    'predicted_prices': prediction.tolist(),
                    'confidence_scores': confidence.tolist() if confidence is not None else None,
                    'timestamp': datetime.now().isoformat()
                }
        else:
            # Single prediction for all data
            prediction, confidence = self.predict(input_data, return_confidence=True)
            results['batch_prediction'] = {
                'predicted_prices': prediction.tolist(),
                'confidence_scores': confidence.tolist() if confidence is not None else None,
                'timestamp': datetime.now().isoformat()
            }
        
        return results
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        info = {
            'model_type': type(self.model).__name__,
            'features_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'metadata': self.metadata,
            'loaded_at': datetime.now().isoformat()
        }
        
        return info
    
    def health_check(self):
        """Perform a health check on the service"""
        try:
            # Create dummy data for testing
            dummy_data = pd.DataFrame([{feature: 1.0 for feature in self.feature_names}])
            
            # Try to make a prediction
            prediction, _ = self.predict(dummy_data)
            
            return {
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'prediction_test': 'passed',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

class PredictionAPI:
    """Simple API wrapper for the prediction service"""
    
    def __init__(self, service):
        self.service = service
    
    def predict_endpoint(self, request_data):
        """Main prediction endpoint"""
        try:
            # Parse input data
            if isinstance(request_data, dict):
                if 'data' in request_data:
                    input_df = pd.DataFrame(request_data['data'])
                else:
                    input_df = pd.DataFrame([request_data])
            else:
                return {"error": "Invalid input format"}
            
            # Make prediction
            if 'Ticker' in input_df.columns:
                results = self.service.batch_predict(input_df)
            else:
                prediction, confidence = self.service.predict(input_df, return_confidence=True)
                results = {
                    'prediction': prediction.tolist(),
                    'confidence': confidence.tolist() if confidence is not None else None,
                    'timestamp': datetime.now().isoformat()
                }
            
            return {'success': True, 'results': results}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def health_endpoint(self):
        """Health check endpoint"""
        return self.service.health_check()
    
    def info_endpoint(self):
        """Model information endpoint"""
        return self.service.get_model_info()

def create_sample_input():
    """Create sample input data for testing"""
    # Load feature names
    try:
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
    except FileNotFoundError:
        print("Feature names file not found. Creating dummy features.")
        feature_names = [f'feature_{i}' for i in range(50)]  # Dummy features
    
    # Create sample data with realistic values
    np.random.seed(42)
    sample_data = {}
    
    for feature in feature_names:
        if 'price' in feature.lower() or 'close' in feature.lower():
            sample_data[feature] = np.random.uniform(100, 300)  # Stock prices
        elif 'volume' in feature.lower():
            sample_data[feature] = np.random.uniform(1000000, 10000000)  # Volume
        elif 'return' in feature.lower():
            sample_data[feature] = np.random.uniform(-0.05, 0.05)  # Returns
        elif 'ratio' in feature.lower():
            sample_data[feature] = np.random.uniform(0.8, 1.2)  # Ratios
        else:
            sample_data[feature] = np.random.uniform(-1, 1)  # General features
    
    return pd.DataFrame([sample_data])

def demo_deployment():
    """Demonstrate the deployment pipeline"""
    print("🚀 Stock Market AI Agent - Deployment Demo")
    print("=" * 60)
    
    # Initialize service
    service = StockPredictionService()
    
    # Load model
    if not service.load_model():
        print("❌ Failed to load model. Make sure models are trained and saved.")
        return
    
    # Create API wrapper
    api = PredictionAPI(service)
    
    # Health check
    print("\\n🏥 Health Check:")
    health = api.health_endpoint()
    print(f"Status: {health['status']}")
    
    # Model info
    print("\\n📋 Model Information:")
    info = api.info_endpoint()
    print(f"Model Type: {info['model_type']}")
    print(f"Features: {info['features_count']}")
    print(f"Training R²: {info['metadata'].get('random_forest_r2', 'N/A')}")
    
    # Sample prediction
    print("\\n🎯 Sample Prediction:")
    sample_input = create_sample_input()
    result = service.predict_single_stock(sample_input, return_details=True)
    print(f"Predicted Price: ${result['predicted_price']:.2f}")
    print(f"Confidence Score: {result.get('confidence_score', 'N/A')}")
    print(f"Model R²: {result.get('training_r2', 'N/A')}")
    
    # Batch prediction demo
    print("\\n📊 Batch Prediction Demo:")
    sample_batch = create_sample_input()
    sample_batch['Ticker'] = 'AAPL'
    
    batch_results = service.batch_predict(sample_batch)
    for ticker, result in batch_results.items():
        prices = result['predicted_prices']
        print(f"{ticker}: ${prices[0]:.2f}")
    
    print("\\n✅ Deployment Demo Complete!")
    print("🌐 Ready for production deployment!")
    
    return service, api

def save_deployment_config():
    """Save deployment configuration"""
    config = {
        "service_name": "stock-price-predictor",
        "version": "1.0.0",
        "model_path": "models/random_forest_model.pkl",
        "feature_names_path": "models/feature_names.pkl",
        "metadata_path": "models/model_metadata.pkl",
        "endpoints": {
            "/predict": "Main prediction endpoint",
            "/health": "Health check endpoint", 
            "/info": "Model information endpoint"
        },
        "deployment_date": datetime.now().isoformat()
    }
    
    with open('models/deployment_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("📁 Deployment configuration saved to models/deployment_config.json")

def main():
    """Main function"""
    # Run deployment demo
    service, api = demo_deployment()
    
    # Save deployment configuration
    save_deployment_config()
    
    return service, api

if __name__ == "__main__":
    main()
