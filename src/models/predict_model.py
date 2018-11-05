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
        self.attribute_ = None
        self.value_ = None
        self.min_impurity = min_impurity
        self.depth = depth
        self.max_depth = max_depth
        self.left_ = None
        self.right_ = None
        self.is_leaf_ = False
        self.label_ = None
        self.confidence_ = None
        self.training_impurity_ = None
    
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
    
    def fit(self, data):
        column_names = data.columns
        x = data[column_names[:-1]]
        y = data[column_names[-1]]
        # Calculate impurity
        impurity = self.impurity(y)
        self.training_impurity_ = impurity
        # Decide if I should split
        if impurity > self.min_impurity and self.depth < self.max_depth:
            # Calculate the split
            self.attribute_, self.value_ = split_attribute(x, y, impurity=self.impurity)
            self.left_ = Node(impurity=self.impurity, min_impurity=self.min_impurity,
                              depth=self.depth+1, max_depth=self.max_depth)
            self.right_ = Node(impurity=self.impurity, min_impurity=self.min_impurity,
                               depth=self.depth+1, max_depth=self.max_depth)
            left_data = data[data[self.attribute_] <= self.value_]
            right_data = data[data[self.attribute_] > self.value_]
            self.left_.fit(left_data)
            self.right_.fit(right_data)
        else:
            self.is_leaf_ = True
            self.label_ = mode(y)
            self.confidence_ = self.impurity(y)
    
    def predict(self, x):
        if self.is_leaf_:
            return self.label_.mode
        else:
            # Determine left or right
            if x[self.attribute_] <= self.value_:
                a = self.left_.predict(x)
                return a
            else:
                a = self.right_.predict(x)
                return a

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
    
    def html_print(self):
        s = "<table border=1 style=\"text-align:center\"><tr style=\"text-align:center\">"
        if self.is_leaf_:
            s += "<td style=\"text-align:center\" bgcolor=\"green\">"
            s += self.__repr__()
            s += "</td>"
        else:
            s += "<td colspan=2 style=\"text-align:center\">"
            s += self.__repr__()
            s += "</td>"
            s += "</tr><tr>"
            s += "<td style=\"text-align:center\" width=50%>"
            s += self.left_.html_print()
            s += "</td><td style=\"text-align:center\" width=50%>"
            s += self.right_.html_print()
            s += "</td>"
        s += "</tr></table>"
        return s
