import numpy as np
from collections import Counter

def dummy_classifier(y_train, n_test, strategy, constant=None):
    """
    Produce baseline predictions of length n_test using the given strategy.
    Returns a Python list of predicted labels.
    """

    counts = Counter(y_train)
    classes = sorted(set(y_train))

    if strategy == "most_frequent":
        best_class = min(classes, key=lambda c:(-counts[c],c))
        return [best_class]*n_test

    elif strategy == "constant":
        return [constant]*n_test
    
    elif strategy == "uniform":
        return [classes[i%len(classes)] for i in range(n_test)]
    
    elif strategy == "stratified":
        exact = [n_test*counts[c]/len(y_train) for c in classes]

        allocations = [int(x) for x in exact]

        remaining = n_test-sum(allocations)

        order = sorted(
            range(len(classes)),
            key=lambda i:(-(exact[i]-allocations[i]),classes[i])
        )

        for i in order[:remaining]:
            allocations[i] += 1
        
        predictions = []
        for c,count in zip(classes,allocations):
            predictions.extend([c]*count)

        return predictions
    
