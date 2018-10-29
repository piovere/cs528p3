import numpy as np


def accuracy(tn, tp, fn, fp):
    return (tn + tp) / (tn + tp + fn + fp)


def true_positive_rate(tp, fn):
    """True positive rate

    Also known as the recall or sensitivity
    """
    return tp / (tp + fn)


def positive_predictive_value(tp, fp):
    """Positive predictive value

    Also known as precision
    """
    return tp / (tp + fp)


def true_negative_rate(tn, fp):
    """True negative rate

    Also known as specificity
    """
    return tn / (tn + fp)


def f1(tp, fn, fp):
    """F1 score
    """
    ppv = positive_predictive_value(tp, fp)
    tpr = true_positive_rate(tp, fn)
    
    return 2 * ppv * tpr / (ppv + tpr)


def confusion(tn, tp, fn, fp):
    return np.array([
        [tn, fp],
        [fn, tp]
    ])
