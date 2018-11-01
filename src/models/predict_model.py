import numpy as np
from src.features.build_features import Scaler
from scipy.spatial.distance import cdist
from scipy.stats import mode
import pandas as pd


class KNN():
    def __init__(self, n, metric='euclidean'):
        self.n = n
        self.xs_ = None
        self.y_ = None
        self.scaler_ = None
        self.metric = metric
    
    def fit(self, x, y):
        self.scaler_ = Scaler()
        self.scaler_.fit(x)
        self.xs_ = self.scaler_.transform(x)
        self.y_ = y

    def predict(self, x):
        # verify training has happened
        if self.xs_ is None:
            raise ValueError("You must train the model first")
        
        # scale x
        xs = self.scaler_.transform(x)

        # calculate the distance from every training vector to every x
        # result will be n_training by n_predict columns
        dists = cdist(self.xs_, xs, self.metric)

        # sort those to find out the closest to each x_predict
        inds = np.argsort(dists, axis=0)[:self.n, :]
        inds = inds.T

        # match those indices to column predictions
        l = mode(self.y_[inds], axis=1)
        labels = l.mode[:, 0, :]

        return labels

class DecisionTreeClassifier():
    def __init__(self, impurity='entropy', max_leaves=10, min_impurity=0.1):
        """Classifier with decision trees

        Implemented here with `pandas.DataFrame`s

        Paramters
        ---------
        impurity : {'entropy', 'gini', 'misclassification'}
        """
        impurity_functions = {
            'entropy': self.entropy,
            'gini': self.gini,
            'misclassification': self.misclassification
        }
        self.impurity = impurity_functions[impurity]
    
    def entropy(self, y):
        """Entropy of a class

        Satisfies equation 9.3 and 9.4 from the text:
        .. math::
            9.3: I = - sum_{i=1}^K p_m^i \log_2 p_m^i \\
            9.4: \phi(p, 1-p) = -p \log_2 p - (1-p) \log_2 (1-p)

        Parameters
        ----------
        y : pandas.DataFrame
            Training data encoded as one-hot vector

        Returns
        -------
        float
            Entropy of the training data
        """
        if y.shape[0] == 0:
            p = 0.0
        else:
            p = np.sum(y) / y.shape[0]
        fp = -p * np.log(p) / np.log(2)
        sp = (1 - p) * np.log(1 - p) / np.log(2)
        return fp - sp
    
    def gini(self, y):
        """Gini index for a class

        Satisfies equation 9.5 from the text:
        .. math::
            9.5: \phi(p, 1-p) = 2 p (1-p)

        Parameters
        ----------

        Returns
        -------

        """
        if y.shape[0] == 0:
            p = 0.0
        else:
            p = np.sum(y) / y.shape[0]
        return 2 * p * (1 - p)

    def misclassification(self, y):
        """Misclassification score of a class

        Implements equation 9.6 from the text:
        .. math::
            9.6: \phi(p, 1-p) = 1 - \mathrm{max}(p, 1-p)

        Parameters
        ----------

        Returns
        -------

        """
        if y.shape[0] == 0:
            p = 0.0
        else:
            p = np.sum(y) / y.shape[0]
        return 1 - np.max([p, 1-p])
    
    def fit(self, x, y):
        """Train the data
        """
        # While all 
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def split_attribute(self, x, y):
        min_entropy = np.finfo(np.float64).max # 1e308 on test system

        xt = x.copy()
        xt['Labels'] = y
        split_column = None
        split_value = None
        for c in x.columns:
            for v in x[c].unique():
                left_split = xt[xt[c] <= v]
                right_split = xt[xt[c] > v]
                left_impurity = self.impurity(left_split['Labels'])
                right_impurity = self.impurity(right_split['Labels'])
                e = left_impurity + right_impurity
                if e < min_entropy:
                    min_entropy = e
                    split_column = c
                    split_value = v
        return split_column, split_value

    @np.vectorize
    def true_negative(self, pred, true):
        if pred == 0 and true == 0:
            return True
        else:
            return False
    
    @np.vectorize
    def false_negative(self, pred, true):
        if pred == 0 and true == 1:
            return True
        else:
            return False

    @np.vectorize
    def true_positive(self, pred, true):
        if pred == 1 and true == 1:
            return True
        else:
            return False

    @np.vectorize
    def false_positive(self, pred, true):
        if pred == 1 and true == 0:
            return True
        else:
            return False

    def confusion(self, pred, true):
        tp = np.sum(self.true_negative(pred, true))
        fp = np.sum(self.false_positive(pred, true))
        tn = np.sum(self.true_negative(pred, true))
        fn = np.sum(self.false_negative(pred, true))

        r = np.array([
            [tp, fp],
            [fn, tn]
        ])
        d = pd.DataFrame(r, index=[1, 0], columns=[1, 0])
        return d

    def accuracy(self, pred, true):
        tp = np.sum(self.true_negative(pred, true))
        fp = np.sum(self.false_positive(pred, true))
        tn = np.sum(self.true_negative(pred, true))
        fn = np.sum(self.false_negative(pred, true))
        return (tn + tp) / (tn + tp + fn + fp)

    def true_positive_rate(self, pred, true):
        """True positive rate

        Also known as the recall or sensitivity
        """
        tp = np.sum(self.true_negative(pred, true))
        fn = np.sum(self.false_negative(pred, true))

        return tp / (tp + fn)

    def positive_predictive_value(self, pred, true):
        """Positive predictive value

        Also known as precision
        """
        tp = np.sum(self.true_negative(pred, true))
        fp = np.sum(self.false_positive(pred, true))

        return tp / (tp + fp)

    def true_negative_rate(self, pred, true):
        """True negative rate

        Also known as specificity
        """
        fp = np.sum(self.false_positive(pred, true))
        tn = np.sum(self.true_negative(pred, true))

        return tn / (tn + fp)

    def f1(self, pred, true):
        """F1 score
        """
        ppv = self.positive_predictive_value(pred, true)
        tpr = self.true_positive_rate(pred, true)
        
        return 2 * ppv * tpr / (ppv + tpr)
