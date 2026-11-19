# --- cell 1 ---
import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    X_bias = np.column_stack((np.ones(X.shape[0]), X))

    parameters, _, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)

    b = float(parameters[0])
    W = parameters[1:]

    return W, b