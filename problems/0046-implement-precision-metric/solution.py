import numpy as np
def precision(y_true, y_pred):
    # Your code here
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))

    if TP+FP == 0:
        return 0.0

    return TP / (TP+FP)