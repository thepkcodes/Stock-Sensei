# Model Comparison Summary

This document summarizes the performance and evaluation results for different machine learning models used in the StockSensei project.

## 1. Random Forest
- **File**: `random_forest_model.pkl`
- **Features**: 183
- **Strengths**: Non-linear predictions, robust to overfitting

## 2. LSTM
- **File**: `enhanced_lstm_model.h5`
- **Features**: Time series-specific
- **Strengths**: Captures temporal dependencies well

## 3. XGBoost
- **File**: `xgboost_model.pkl`
- **Features**: 183
- **Strengths**: Fast and efficient, handles missing data well

## Advanced Ensemble Models

### Voting Ensemble
- **File**: Did not retain due to size
- **Components**: Random Forest, XGBoost, Ridge Regression
- **Strengths**: Combines multiple models for robust predictions

### Bagging Ensemble
- **File**: Did not retain due to size
- **Method**: Combines multiple Random Forest models

## Performance Metrics

| Model             | RMSE | MAE  | R²   |
|-------------------|------|------|------|
| Random Forest     | ...  | ...  | ...  |
| LSTM              | ...  | ...  | ...  |
| XGBoost           | ...  | ...  | ...  |
| Voting Ensemble   | ...  | ...  | ...  |
| Bagging Ensemble  | ...  | ...  | ...  |

## Conclusion

- **Best Overall Performer**: [Model Name] achieved the best results with an R² of [Value].
- **Recommended for Live App**: Random Forest is used for the live app due to its balance of performance and complexity.

This comprehensive comparison illustrates the capability of different models and supports the choice of models for different use cases within the project.
