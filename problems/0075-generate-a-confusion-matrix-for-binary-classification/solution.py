
from collections import Counter

def confusion_matrix(data): #[y_true, y_pred]
    # Implement the function here
    counts = Counter(tuple(item) for item in data)

    TP = counts[(1, 1)]
    FN = counts[(1, 0)]
    FP = counts[(0, 1)]
    TN = counts[(0, 0)]

    return [[TP, FN], [FP, TN]]
