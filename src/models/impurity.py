import numpy as np


def entropy(y):
    if y.shape[0] == 0:
        p = 0.0
    else:
        p = np.sum(y) / y.shape[0]
    
    if p == 0.0:
        fp = 0.0
    else:
        fp = -p * np.log2(p)
    
    if p == 1.0:
        sp = 0.0
    else:
        sp = (1 - p) * np.log2(1 - p)
    
    return fp - sp

def gini(y):
    if y.shape[0] == 0:
        p = 0.0
    else:
        p = np.sum(y) / y.shape[0]

    return 2 * p * (1 - p)

def misclassification(y):
    if y.shape[0] == 0:
        p = 0.0
    else:
        p = np.sum(y) / y.shape[0]
    
    return 1 - np.max([p, 1-p])
