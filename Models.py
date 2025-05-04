from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_linear_regression(X, y, reg_type='ridge', alpha=1.0):
    """
    Train a regularized regression model to prevent overfitting.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    alpha : float
        Regularization strength (higher values = more regularization)
    
    Returns:
    --------
    model : trained model
    X_test : test features
    y_test : test targets
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize model based on regularization type
    if reg_type == 'lasso':
        model = Lasso(alpha=alpha, random_state=42)
    else:  # default to ridge regression
        model = Ridge(alpha=alpha, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Print feature importance with column names
    print("\nFeature Importances:")
    for feature_name, coef in zip(X.columns, abs(model.coef_)):
        print(f"{feature_name}: {coef:.4f}")

    return model, X_test, y_test

