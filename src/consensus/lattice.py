import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class MeetConsensus(BaseEstimator, ClusterMixin):

    def fit_predict(self, X, y=None):
        valid = np.all(X >= 0, axis=1)
        barcodes, labels = np.unique(X[valid], return_inverse=True, axis=0)
        out = np.full(X.shape[0], -1)
        out[valid] = labels
        return out
