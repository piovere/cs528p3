import numpy as np


class Scaler():
    def __init__(self):
        self.means_ = None
        self.stds_ = None

    def fit(self, x):
        self.means_ = np.mean(x, axis=0)
        self.stds_ = np.std(x, axis=0)
        return self
    
    def transform(self, x):
        return (x - self.means_) / self.stds_
