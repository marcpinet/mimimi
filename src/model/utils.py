import numpy as np
from sklearn.metrics import roc_curve


def find_youden_threshold(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold
