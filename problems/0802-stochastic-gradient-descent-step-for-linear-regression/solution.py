import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)

    n_samples = X.shape[0]

    for i in range(n_iter):

        idx = i%n_samples

        x_i = X[idx]
        y_i = y[idx]

        y_pred = x_i@weights
        error = y_pred-y_i

        gradient = 2*error*x_i

        weights = weights - learning_rate*gradient

    return weights.tolist()
