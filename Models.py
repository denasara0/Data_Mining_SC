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
    
    # Print feature importance
    print("\nFeature Importances:")
    for idx, coef in enumerate(abs(model.coef_)):
        print(f"Feature {idx}: {coef:.4f}")
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest Set Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.3f}")
    print(f"Root Mean Squared Error: {np.sqrt(mse):.2f}")
    
    return mse, r2, y_pred
