import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    y = np.asarray(y)
    n_samples = len(y)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    if n_train == 0:
        raise ValueError("训练集至少需要一个样本")

    training_mean = float(np.mean(y))
    test_predictions = np.full(len(test_idx), training_mean)

    return test_predictions, train_idx, val_idx, test_idx
