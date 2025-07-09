import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelOptimizer:
    """Advanced model optimization with hyperparameter tuning and time-series validation"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.best_models = {}
        self.optimization_results = {}
        
    def load_and_prepare_data(self, target_col='target_price_1d'):
        """Load and prepare data with proper time series handling"""
        print("Loading and preparing data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Clean data
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.dropna(inplace=True)
        
        # Sort by date for time series
        self.data = self.data.sort_values(['Ticker', 'Date'])
        self.data = self.data.dropna(subset=[target_col])
        
        # Prepare features and target
        drop_cols = ['Date', 'Ticker']
        target_cols = [col for col in self.data.columns if col.startswith('target_')]
        drop_cols.extend([col for col in target_cols if col != target_col])
        
        X = self.data.drop(columns=drop_cols)
        y = self.data[target_col]
        
        print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def time_series_validation(self, model, X, y, n_splits=5):
        """Perform time series cross validation"""
        print(f"Performing time series cross-validation with {n_splits} splits...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Fit model on training fold
            model.fit(X_train_fold, y_train_fold)
            
            # Predict on validation fold
            y_pred = model.predict(X_val_fold)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            mae = mean_absolute_error(y_val_fold, y_pred)
            r2 = r2_score(y_val_fold, y_pred)
            
            scores.append({'fold': fold + 1, 'rmse': rmse, 'mae': mae, 'r2': r2})
            print(f"  Fold {fold + 1}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        # Calculate average scores
        avg_scores = {
            'rmse_mean': np.mean([s['rmse'] for s in scores]),
            'rmse_std': np.std([s['rmse'] for s in scores]),
            'mae_mean': np.mean([s['mae'] for s in scores]),
            'mae_std': np.std([s['mae'] for s in scores]),
            'r2_mean': np.mean([s['r2'] for s in scores]),
            'r2_std': np.std([s['r2'] for s in scores])
        }
        
        print(f"Average CV scores:")
        print(f"  RMSE: {avg_scores['rmse_mean']:.4f} ± {avg_scores['rmse_std']:.4f}")
        print(f"  MAE: {avg_scores['mae_mean']:.4f} ± {avg_scores['mae_std']:.4f}")
        print(f"  R²: {avg_scores['r2_mean']:.4f} ± {avg_scores['r2_std']:.4f}")
        
        return scores, avg_scores
    
    def optimize_random_forest(self, X, y, cv_splits=3):
        """Optimize Random Forest hyperparameters"""
        print("🌲 Optimizing Random Forest hyperparameters...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Create base model
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Time series cross validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        # Randomized search for efficiency
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=20,  # Number of parameter combinations to try
            cv=tscv,
            scoring='r2',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the search
        random_search.fit(X, y)
        
        print(f"Best Random Forest parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        
        # Store results
        self.best_models['RandomForest'] = random_search.best_estimator_
        self.optimization_results['RandomForest'] = {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_results': random_search.cv_results_
        }
        
        return random_search.best_estimator_
    
    def optimize_xgboost(self, X, y, cv_splits=3):
        """Optimize XGBoost hyperparameters"""
        print("🚀 Optimizing XGBoost hyperparameters...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0.1, 1, 2]
        }
        
        # Create base model
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
        
        # Time series cross validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        # Randomized search for efficiency
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=20,  # Number of parameter combinations to try
            cv=tscv,
            scoring='r2',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the search
        random_search.fit(X, y)
        
        print(f"Best XGBoost parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        
        # Store results
        self.best_models['XGBoost'] = random_search.best_estimator_
        self.optimization_results['XGBoost'] = {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_results': random_search.cv_results_
        }
        
        return random_search.best_estimator_
    
    def evaluate_optimized_models(self, X, y):
        """Evaluate optimized models with time series validation"""
        print("\\n📊 Evaluating Optimized Models")
        print("=" * 50)
        
        results = {}
        
        for model_name, model in self.best_models.items():
            print(f"\\nEvaluating {model_name}...")
            scores, avg_scores = self.time_series_validation(model, X, y, n_splits=5)
            results[model_name] = {
                'individual_scores': scores,
                'average_scores': avg_scores
            }
        
        # Compare models
        comparison_df = pd.DataFrame([
            {
                'Model': name,
                'RMSE': results[name]['average_scores']['rmse_mean'],
                'RMSE_std': results[name]['average_scores']['rmse_std'],
                'MAE': results[name]['average_scores']['mae_mean'],
                'MAE_std': results[name]['average_scores']['mae_std'],
                'R²': results[name]['average_scores']['r2_mean'],
                'R²_std': results[name]['average_scores']['r2_std']
            }
            for name in results.keys()
        ])
        
        print("\\n🏆 Model Comparison (Time Series CV):")
        print(comparison_df.round(4))
        
        # Find best model
        best_model_name = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
        print(f"\\n🥇 Best Model: {best_model_name}")
        
        return results, comparison_df
    
    def save_optimized_models(self):
        """Save optimized models and results"""
        print("\\n💾 Saving optimized models...")
        
        # Save models
        for model_name, model in self.best_models.items():
            model_path = f'models/optimized_{model_name.lower()}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {model_name} to {model_path}")
        
        # Save optimization results
        results_path = 'models/optimization_results.json'
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for model_name, results in self.optimization_results.items():
            json_results[model_name] = {
                'best_params': results['best_params'],
                'best_score': float(results['best_score'])
            }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"Saved optimization results to {results_path}")
    
    def run_complete_optimization(self, target_col='target_price_1d'):
        """Run the complete optimization pipeline"""
        print("🚀 Starting Advanced Model Optimization Pipeline")
        print("=" * 60)
        
        # Load and prepare data
        X, y = self.load_and_prepare_data(target_col)
        
        # Optimize Random Forest
        print("\\n" + "="*60)
        self.optimize_random_forest(X, y)
        
        # Optimize XGBoost
        print("\\n" + "="*60)
        self.optimize_xgboost(X, y)
        
        # Evaluate all optimized models
        print("\\n" + "="*60)
        results, comparison_df = self.evaluate_optimized_models(X, y)
        
        # Save models and results
        self.save_optimized_models()
        
        # Save comparison results
        comparison_df.to_csv('models/optimized_model_comparison.csv', index=False)
        
        print("\\n✅ Optimization Complete!")
        print("🏆 Your models are now optimized and ready for production!")
        
        return self.best_models, results, comparison_df

    def advanced_feature_selection(self, X, y, model_type='random_forest'):
        """Perform advanced feature selection using multiple methods"""
        print(f"\n🔍 Performing advanced feature selection with {model_type}...")
        
        from sklearn.feature_selection import SelectKBest, f_regression, RFE
        from sklearn.ensemble import RandomForestRegressor
        import xgboost as xgb
        
        results = {}
        
        # 1. Statistical feature selection (F-test)
        print("  📊 Statistical feature selection...")
        k_best = SelectKBest(score_func=f_regression, k='all')
        k_best.fit(X, y)
        statistical_scores = pd.DataFrame({
            'feature': X.columns,
            'f_score': k_best.scores_,
            'p_value': k_best.pvalues_
        }).sort_values('f_score', ascending=False)
        
        # Select top features based on p-value
        significant_features = statistical_scores[statistical_scores['p_value'] < 0.05]['feature'].tolist()
        results['statistical'] = {
            'selected_features': significant_features,
            'scores': statistical_scores
        }
        print(f"    Selected {len(significant_features)} statistically significant features")
        
        # 2. Model-based feature importance
        print("  🌲 Model-based feature importance...")
        if model_type == 'random_forest':
            temp_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            temp_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        temp_model.fit(X, y)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': temp_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top 80% of features by importance
        importance_threshold = feature_importance['importance'].quantile(0.2)
        important_features = feature_importance[feature_importance['importance'] > importance_threshold]['feature'].tolist()
        results['importance'] = {
            'selected_features': important_features,
            'scores': feature_importance
        }
        print(f"    Selected {len(important_features)} features based on importance")
        
        # 3. Recursive Feature Elimination
        print("  🔄 Recursive feature elimination...")
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        n_features_to_select = min(100, X.shape[1] // 2)  # Select half of features or 100, whichever is smaller
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=10)
        rfe.fit(X, y)
        rfe_features = X.columns[rfe.support_].tolist()
        results['rfe'] = {
            'selected_features': rfe_features,
            'ranking': rfe.ranking_
        }
        print(f"    Selected {len(rfe_features)} features using RFE")
        
        # 4. Combine all methods
        print("  🤝 Combining feature selection methods...")
        combined_features = list(set(significant_features) & set(important_features) & set(rfe_features))
        if len(combined_features) < 20:  # Ensure minimum features
            combined_features = list(set(significant_features) | set(important_features[:50]))
        
        results['combined'] = {
            'selected_features': combined_features,
            'count': len(combined_features)
        }
        print(f"    Final combined selection: {len(combined_features)} features")
        
        return results
    
    def ensemble_modeling(self, X, y, feature_selection_results=None):
        """Create ensemble models with different combinations"""
        print("\n🎭 Creating ensemble models...")
        
        from sklearn.ensemble import VotingRegressor, BaggingRegressor
        from sklearn.linear_model import Ridge
        
        # Use selected features if available
        if feature_selection_results and 'combined' in feature_selection_results:
            selected_features = feature_selection_results['combined']['selected_features']
            X_selected = X[selected_features]
            print(f"Using {len(selected_features)} selected features for ensemble")
        else:
            X_selected = X
            print(f"Using all {X.shape[1]} features for ensemble")
        
        ensemble_models = {}
        
        # 1. Voting Regressor (combines RF and XGB)
        print("  🗳️ Creating voting regressor...")
        rf_estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        xgb_estimator = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        ridge_estimator = Ridge(alpha=1.0, random_state=42)
        
        voting_regressor = VotingRegressor([
            ('rf', rf_estimator),
            ('xgb', xgb_estimator),
            ('ridge', ridge_estimator)
        ])
        
        ensemble_models['voting'] = voting_regressor
        
        # 2. Bagging with Random Forest
        print("  🎒 Creating bagged ensemble...")
        bagging_regressor = BaggingRegressor(
            estimator=RandomForestRegressor(n_estimators=50, random_state=42),
            n_estimators=10,
            random_state=42,
            n_jobs=-1
        )
        ensemble_models['bagging'] = bagging_regressor
        
        # Evaluate ensemble models
        ensemble_results = {}
        for name, model in ensemble_models.items():
            print(f"  📊 Evaluating {name} ensemble...")
            scores, avg_scores = self.time_series_validation(model, X_selected, y, n_splits=3)
            ensemble_results[name] = {
                'model': model,
                'scores': scores,
                'avg_scores': avg_scores,
                'features_used': X_selected.columns.tolist() if hasattr(X_selected, 'columns') else list(range(X_selected.shape[1]))
            }
        
        return ensemble_models, ensemble_results
    
    def advanced_model_validation(self, X, y):
        """Perform advanced validation with multiple strategies"""
        print("\n🔬 Advanced Model Validation")
        print("=" * 50)
        
        from sklearn.model_selection import cross_val_score, cross_validate
        
        validation_results = {}
        
        for model_name, model in self.best_models.items():
            print(f"\nValidating {model_name}...")
            
            # 1. Time Series Cross Validation (already implemented)
            ts_scores, ts_avg = self.time_series_validation(model, X, y, n_splits=5)
            
            # 2. Rolling Window Validation
            print("  📅 Rolling window validation...")
            window_size = len(X) // 6  # 6 windows
            rolling_scores = []
            
            for i in range(5):
                start_idx = i * window_size
                end_idx = start_idx + window_size * 2
                if end_idx > len(X):
                    break
                    
                train_end = start_idx + window_size
                X_train_window = X.iloc[start_idx:train_end]
                y_train_window = y.iloc[start_idx:train_end]
                X_test_window = X.iloc[train_end:end_idx]
                y_test_window = y.iloc[train_end:end_idx]
                
                model.fit(X_train_window, y_train_window)
                y_pred_window = model.predict(X_test_window)
                
                window_r2 = r2_score(y_test_window, y_pred_window)
                rolling_scores.append(window_r2)
            
            rolling_avg = np.mean(rolling_scores) if rolling_scores else 0
            
            # 3. Out-of-time validation (using last 20% as holdout)
            print("  ⏰ Out-of-time validation...")
            split_idx = int(len(X) * 0.8)
            X_train_oot = X.iloc[:split_idx]
            y_train_oot = y.iloc[:split_idx]
            X_test_oot = X.iloc[split_idx:]
            y_test_oot = y.iloc[split_idx:]
            
            model.fit(X_train_oot, y_train_oot)
            y_pred_oot = model.predict(X_test_oot)
            oot_r2 = r2_score(y_test_oot, y_pred_oot)
            
            validation_results[model_name] = {
                'time_series_cv': ts_avg,
                'rolling_window': {'scores': rolling_scores, 'avg': rolling_avg},
                'out_of_time': oot_r2
            }
            
            print(f"    Time Series CV R²: {ts_avg['r2_mean']:.4f} ± {ts_avg['r2_std']:.4f}")
            print(f"    Rolling Window R²: {rolling_avg:.4f}")
            print(f"    Out-of-Time R²: {oot_r2:.4f}")
        
        return validation_results
    
    def run_enhanced_optimization(self, target_col='target_price_1d'):
        """Run the enhanced optimization pipeline with all advanced features"""
        print("🚀 Starting Enhanced Model Optimization Pipeline")
        print("=" * 70)
        
        # Load and prepare data
        X, y = self.load_and_prepare_data(target_col)
        
        # 1. Basic optimization (existing)
        print("\n" + "="*70)
        print("PHASE 1: BASIC HYPERPARAMETER OPTIMIZATION")
        print("="*70)
        self.optimize_random_forest(X, y)
        self.optimize_xgboost(X, y)
        
        # 2. Advanced feature selection
        print("\n" + "="*70)
        print("PHASE 2: ADVANCED FEATURE SELECTION")
        print("="*70)
        feature_selection_results = self.advanced_feature_selection(X, y)
        
        # 3. Ensemble modeling
        print("\n" + "="*70)
        print("PHASE 3: ENSEMBLE MODELING")
        print("="*70)
        ensemble_models, ensemble_results = self.ensemble_modeling(X, y, feature_selection_results)
        
        # Add ensemble models to best_models
        for name, model in ensemble_models.items():
            self.best_models[f'Ensemble_{name}'] = model
        
        # 4. Advanced validation
        print("\n" + "="*70)
        print("PHASE 4: ADVANCED MODEL VALIDATION")
        print("="*70)
        validation_results = self.advanced_model_validation(X, y)
        
        # 5. Final evaluation with all models
        print("\n" + "="*70)
        print("PHASE 5: COMPREHENSIVE MODEL COMPARISON")
        print("="*70)
        all_results, comparison_df = self.evaluate_optimized_models(X, y)
        
        # Save everything
        self.save_optimized_models()
        
        # Enhanced results summary
        enhanced_results = {
            'basic_optimization': self.optimization_results,
            'feature_selection': feature_selection_results,
            'ensemble_results': ensemble_results,
            'validation_results': validation_results,
            'final_comparison': comparison_df
        }
        
        # Save enhanced results
        with open('models/enhanced_optimization_results.json', 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = {}
            for key, value in enhanced_results.items():
                if key == 'final_comparison':
                    json_results[key] = value.to_dict()
                elif key == 'feature_selection':
                    json_results[key] = {
                        method: {
                            'selected_features': data['selected_features'],
                            'count': len(data['selected_features'])
                        } for method, data in value.items() if 'selected_features' in data
                    }
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2, default=str)
        
        comparison_df.to_csv('models/enhanced_model_comparison.csv', index=False)
        
        print("\n" + "="*70)
        print("✅ ENHANCED OPTIMIZATION COMPLETE!")
        print("="*70)
        print("🏆 All models optimized with advanced techniques!")
        print("📊 Results saved to models/enhanced_optimization_results.json")
        print("📈 Comparison saved to models/enhanced_model_comparison.csv")
        
        return self.best_models, enhanced_results, comparison_df

def main():
    """Main function"""
    optimizer = ModelOptimizer(data_path='data/processed_features.csv')
    
    # Run enhanced optimization
    best_models, enhanced_results, comparison = optimizer.run_enhanced_optimization()
    
    return best_models, enhanced_results, comparison

if __name__ == "__main__":
    main()
