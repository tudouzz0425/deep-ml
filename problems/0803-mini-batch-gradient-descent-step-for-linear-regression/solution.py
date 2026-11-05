import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    X = np.asarray(X,dtype=float) #(N,D)
    y = np.asarray(y,dtype=float) #(N,)
    weights = np.asarray(weights,dtype=float).copy() #(D,)
    X_batch = X[batch_indices] #(m,D)
    y_batch = y[batch_indices] #(m,)
    m = len(batch_indices)

    predictions = X_batch @ weights + bias #(m,)
    errors = predictions - y_batch

    grad_weights = (2/m) * X_batch.T @ errors
    grad_bias = (2/m) * np.sum(errors)

    weights = weights - lr * grad_weights
    bias = bias - lr * grad_bias

    return np.concatenate([weights,[bias]])








