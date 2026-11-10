import numpy as np

def f_score(y_true, y_pred, beta):
    """
    Calculate F-Score for a binary classification task.

    :param y_true: Numpy array of true labels
    :param y_pred: Numpy array of predicted labels
    :param beta: The weight of precision in the harmonic mean
    :return: F-Score rounded to three decimal places
    """
    # 计算 TP、FP、FN
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # 计算 Precision
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0

    # 计算 Recall
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0

    # 计算 F-beta Score
    beta2 = beta ** 2
    denominator = beta2 * precision + recall

    if denominator == 0:
        return 0.0

    score = (1 + beta2) * precision * recall / denominator

    return round(score, 3)
