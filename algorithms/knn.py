import numpy as np

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        dists = np.linalg.norm(self.X_train - x, axis=1)
        k_idx = np.argsort(dists)[:self.k]
        k_labels = self.y_train[k_idx]
        vals, counts = np.unique(k_labels, return_counts=True)
        return vals[np.argmax(counts)]

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)
