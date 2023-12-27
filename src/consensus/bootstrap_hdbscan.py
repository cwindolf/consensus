import warnings

import numpy as np
from hdbscan import HDBSCAN
from sklearn.base import BaseEstimator, TransformerMixin


class BootstrapHDBSCAN(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_subsets=20,
        subset_fraction=0.75,
        hdsban_kwargs=None,
        random_state=None,
    ):
        self.n_subsets = n_subsets
        self.subset_fraction = subset_fraction
        self.hdsban_kwargs = hdsban_kwargs
        self.random_state = random_state

    def fit(self, X, y=None):
        N, p = X.shape
        self.N_ = N

        rg = np.random.default_rng(self.random_state)
        subset_S = int(np.ceil(self.subset_fraction * self.n_subsets))
        if subset_S == self.n_subsets:
            warnings.warn("This subset_fraction has no effect (it's too large, or n_subsets is too small)")
        self.mask_ = np.zeros((N, self.n_subsets), dtype=bool)
        for i in range(N):
            self.mask_[i, rg.choice(self.n_subsets, size=subset_S, replace=False)] = 1
        # subset_size = int(np.ceil(self.subset_fraction * N))
        self.masks_ = [np.flatnonzero(self.mask_[:, s]) for s in range(self.n_subsets)]

        hdbscan_kwargs = self.hdsban_kwargs or {}
        self.clusterers_ = [HDBSCAN(**hdbscan_kwargs) for _ in range(self.n_subsets)]

        for clusterer, mask in zip(self.clusterers_, self.masks_):
            clusterer.fit(X[mask])

        y = np.full((self.N_, self.n_subsets), -1)
        for s, (clusterer, mask) in enumerate(zip(self.clusterers_, self.masks_)):
            y[mask, s] = clusterer.labels_
        self.y_ = y

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.y_

    def transform(self, X, y=None):
        return self.fit_transform(X)
