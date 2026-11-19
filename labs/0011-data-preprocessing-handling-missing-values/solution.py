# --- cell 1 ---
import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = X.copy()
    
    for col in range(X_clean.shape[1]):
        column = X_clean[:, col]
        missing = np.isnan(column)
        observed = column[~missing]

        if observed.size > 0:
            fill_value = np.median(observed)
        else:
            fill_value = 0.0

        X_clean[missing, col] = fill_value

    return X_clean