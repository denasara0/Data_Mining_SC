import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from Models import train_linear_regression, evaluate_model
from Data_loading import loader
from Data_cleaning import encode_features, scale_features

def load_and_prepare_data():
    # Load the data
    df = loader("medical_costs.csv")
    
    # Encode categorical features
    df = encode_features(df)
    
    # Prepare features and target
    X = df.drop('Medical Cost', axis=1)
    y = df['Medical Cost']
    
    # Scale the features
    X = scale_features(X, X.columns)
    
    return X, y

@pytest.mark.parametrize("reg_type,alpha", [
    ('ridge', 0.1),
    ('ridge', 1.0),
    ('ridge', 10.0),
    ('lasso', 0.1),
    ('lasso', 1.0),
    ('lasso', 10.0)
])
def test_regularized_model(reg_type, alpha):
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Train model with different regularization settings
    model, X_test, y_test = train_linear_regression(X, y, reg_type=reg_type, alpha=alpha)
    
    # Evaluate model
    mse, r2, y_pred = evaluate_model(model, X_test, y_test)
    
    # Basic assertions
    assert model is not None, "Model should not be None"
    assert hasattr(model, 'coef_'), "Model should have coefficients"
    assert hasattr(model, 'intercept_'), "Model should have an intercept"
    
    # Test predictions
    assert y_pred.shape == y_test.shape, "Predictions should have same shape as test data"
    assert not np.isnan(y_pred).any(), "Predictions should not contain NaN values"
    assert not np.isinf(y_pred).any(), "Predictions should not contain infinite values"
    assert (y_pred >= 0).all(), "Medical costs predictions should be non-negative"
    
    # Test performance
    assert mse >= 0, "MSE should be non-negative"
    assert r2 <= 1, "R2 score should be less than or equal to 1"
    assert r2 > 0.3, "R2 score should be better than random guessing"
    
    # Print regularization effect
    print(f"\nRegularization effect ({reg_type}, alpha={alpha}):")
    print(f"Number of features with coefficient > 0.01: {sum(abs(model.coef_) > 0.01)}")
    print(f"Average coefficient magnitude: {np.mean(abs(model.coef_)):.4f}")

def test_compare_regularization():
    """Compare performance of different regularization settings"""
    X, y = load_and_prepare_data()
    results = {}
    
    for reg_type in ['ridge', 'lasso']:
        for alpha in [0.1, 1.0, 10.0]:
            # Train and evaluate model
            model, X_test, y_test = train_linear_regression(X, y, reg_type=reg_type, alpha=alpha)
            mse, r2, _ = evaluate_model(model, X_test, y_test)
            
            key = f"{reg_type}_{alpha}"
            results[key] = {
                'MSE': mse,
                'R2': r2,
                'NonZero': sum(abs(model.coef_) > 0.01)
            }
    
    # Print comparison
    print("\nRegularization Comparison:")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  MSE = {metrics['MSE']:.2f}")
        print(f"  R² = {metrics['R2']:.3f}")
        print(f"  Features used = {metrics['NonZero']}")
    
    # Assert that at least one model performs well
    best_r2 = max(result['R2'] for result in results.values())
    assert best_r2 > 0.4, "At least one model should achieve R² > 0.4" 
