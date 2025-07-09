#!/usr/bin/env python3
"""
Script to inspect .pkl files in the models directory
"""

import pickle
import os
import sys
import json
import numpy as np

def inspect_pkl_file(file_path):
    """Load and inspect a pickle file"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {file_path}")
    print(f"{'='*60}")
    
    try:
        # Load the pickle file
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        
        # Get basic info
        print(f"Type: {type(obj)}")
        print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        
        # If it's a scikit-learn model
        if hasattr(obj, 'get_params'):
            print(f"\nModel Type: {obj.__class__.__name__}")
            print(f"Parameters: {json.dumps(obj.get_params(), indent=2, default=str)}")
            
            if hasattr(obj, 'n_features_in_'):
                print(f"Number of features: {obj.n_features_in_}")
            
            if hasattr(obj, 'feature_importances_'):
                print(f"Has feature importances: Yes ({len(obj.feature_importances_)} features)")
        
        # If it's a list (like feature names)
        elif isinstance(obj, list):
            print(f"List length: {len(obj)}")
            print(f"First 5 items: {obj[:5]}")
            print(f"Last 5 items: {obj[-5:]}")
        
        # If it's a dict
        elif isinstance(obj, dict):
            print(f"Dictionary keys: {list(obj.keys())}")
            for key, value in obj.items():
                print(f"  {key}: {type(value)}")
        
        # If it's a numpy array
        elif isinstance(obj, np.ndarray):
            print(f"Array shape: {obj.shape}")
            print(f"Array dtype: {obj.dtype}")
        
        else:
            print(f"Object info: {str(obj)[:200]}...")
            
    except Exception as e:
        print(f"Error loading file: {e}")

def main():
    # List all .pkl files in models directory
    models_dir = 'models'
    pkl_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    print(f"Found {len(pkl_files)} .pkl files in {models_dir}/")
    
    # Inspect each file
    for pkl_file in sorted(pkl_files):
        file_path = os.path.join(models_dir, pkl_file)
        inspect_pkl_file(file_path)
    
    # Example of how to load and use a model
    print("\n" + "="*60)
    print("Example: How to load and use a model in your code:")
    print("="*60)
    print("""
import pickle

# Load the model
with open('models/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature names
with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Now you can use the model for predictions
# predictions = model.predict(X_test)
    """)

if __name__ == "__main__":
    main()
