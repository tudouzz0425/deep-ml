import numpy as np

def precision_recall_at_threshold(y_true, y_scores, threshold):
    """
    Compute precision and recall at a given decision threshold.

    Args:
        y_true: list/array of true binary labels (0 or 1)
        y_scores: list/array of predicted scores in [0, 1]
        threshold: float, classification threshold (predict positive if score >= threshold)

    Returns:
        [precision, recall] as a list of two floats rounded to 4 decimals.
    """
    y_true = np.asarray(y_true)
    y_predict = np.asarray(y_scores) >= threshold
    TP = np.sum((y_predict == 1) & (y_true == 1))
    FP = np.sum((y_predict == 1) & (y_true == 0))
    FN = np.sum((y_predict == 0) & (y_true == 1))

    if TP+FP == 0:
        precision = 0.0
    else:
        precision = round(TP/(TP+FP),4)
    
    if TP+FN == 0:
        recall = 0.0
    else:
        recall = round(TP/(TP+FN),4)

    return [precision,recall]