import numpy as np
import pandas as pd
from scipy.stats import mode
from src.models.impurity import entropy, gini, misclassification


class DecisionTree():
    def __init__(self, impurity=entropy, depth=0, max_depth=10, 
                 min_impurity=0.001):
        self.impurity = impurity
        self.depth = depth
        self.max_depth = max_depth
        self.min_impurity = min_impurity

        # Set during training
        self.left_ = None
        self.right_ = None
        self.column_ = None
        self.value_ = None

        # Set for leaves only
        self.is_leaf_ = False
        self.class_ = None
    
    def fit(self, data):
        y = data[data.columns[-1]]

        imp = self.impurity(y)
        print(f"Impurity {imp:0.2f} at depth {self.depth}")

        if self.depth >= self.max_depth:
            self.is_leaf_ = True
            print("I Found a leaf")
            m = mode(y, axis=None).mode
            self.class_ = m
        elif imp < self.min_impurity:
            self.is_leaf_ = True
            print("Branch is sufficiently pure")
        else:
            # make two branches
            self.left_ = DecisionTree(impurity=self.impurity, 
                                      depth=self.depth+1, 
                                      max_depth=self.max_depth, 
                                      min_impurity=self.min_impurity)
            self.right_ = DecisionTree(impurity=self.impurity, 
                                      depth=self.depth+1, 
                                      max_depth=self.max_depth, 
                                      min_impurity=self.min_impurity)
            
            split_column, split_value = self.split(data)
            self.column_ = split_column
            self.value_ = split_value

            left_data = data[data[split_column] < split_value].copy()
            right_data = data[data[split_column] > split_value].copy()

            self.left_ = self.left_.fit(left_data)
            self.right_ = self.right_.fit(right_data)
        
        return self
    
    def predict(self, data):
        y = []
        for i in data.index:
            y.append(self.s_predict(data.loc[i])[0])

    def s_predict(self, row):
        if self.is_leaf_:
            pd.Series()
            return self.class_
        else:
            # Determine if left or right
            if (row[self.column_] < self.value_):
                return self.left_.s_predict(row)
            else:
                return self.right_.s_predict(row)

    def split(self, data):
        min_imp = np.finfo(np.float64).max # 1e308 on test system

        split_column = None
        split_value = None

        x = data.copy()
        yc = x.columns[-1]

        for c in x.columns[:-1]:
            for v in x[c].unique():
                left_split = x[x[c] < v + 0.5].copy()
                right_split = x[x[c] > v + 0.5].copy()

                left_imp = self.impurity(left_split[yc])
                right_imp = self.impurity(right_split[yc])

                e = left_imp + right_imp

                if e < min_imp:
                    min_imp = e
                    split_column = c
                    split_value = v + 0.5

        return split_column, split_value
    
    def accuracy(self, data):
        r = self.confusion(data)
        return (r[0, 0] + r[1, 1]) / np.sum(r)
    
    def tpr(self, data):
        r = self.confusion(data)
        return r[1, 1] / (r[1, 1] + r[1, 0])
    
    def ppv(self, data):
        r = self.confusion(data)
        return r[1, 1] / (r[1, 1] + r[0, 1])
    
    def tnr(self, data):
        r = self.confusion(data)
        return r[0, 0] / (r[0, 0] + r[0, 1])
    
    def f1_score(self, data):
        num = 2 * self.ppv(data) * self.tpr(data)
        den = self.ppv(data) + self.tpr(data)
        return num / den

    @np.vectorize
    def true_negative(self, data):
        y = data[data.columns[-1]]
        yp = self.predict(data[data.columns[:-1]])
        
        if yp == 0 and y == 0:
            return True
        else:
            return False
    
    @np.vectorize
    def false_negative(self, data):
        y = data[data.columns[-1]]
        yp = self.predict(data[data.columns[:-1]])
        
        if yp == 0 and y == 1:
            return True
        else:
            return False

    @np.vectorize
    def true_positive(self, data):
        y = data[data.columns[-1]]
        yp = self.predict(data[data.columns[:-1]])
        
        if yp == 1 and y == 1:
            return True
        else:
            return False

    @np.vectorize
    def false_positive(self, data):
        y = data[data.columns[-1]]
        yp = self.predict(data[data.columns[:-1]])
        
        if yp == 1 and y == 0:
            return True
        else:
            return False

    def confusion(self, data):
        tp = np.sum(self.true_negative(data))
        fp = np.sum(self.false_positive(data))
        tn = np.sum(self.true_negative(data))
        fn = np.sum(self.false_negative(data))

        r = np.array([
            [tn, fp],
            [fn, tp]
        ])

        return r

    def print_report(self, data):
        print("Selected Model Performance")
        print("----------------------------")
        print(f"Accuracy\t\t{self.accuracy(data):0.2f}")
        print(f"True Positive Rate\t{self.tpr(data):0.2f}")
        print(f"Precision\t\t{self.ppv(data):0.2f}")
        print(f"Specificity\t\t{self.tnr(data):0.2f}")
        print(f"F1 Score\t\t{self.f1_score(data):0.2f}")

    def __repr__(self):
        if self.is_leaf_:
            return f"Leaf valued {self.class_}"
        else:
            return f"Node split on {self.column_} at {self.value_}"
