import numpy as np

class DecisionTree:
    def __init__(self, max_depth=10, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None

    def _gini(self, y):
        classes = np.unique(y)
        n = len(y)
        return 1 - sum((np.sum(y == c) / n) ** 2 for c in classes)

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1, None, None
        parent_gini = self._gini(y)
        n = len(y)
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left = y[X[:, feat] <= t]
                right = y[X[:, feat] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = parent_gini - (len(left)/n * self._gini(left) + len(right)/n * self._gini(right))
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, t
        return best_feat, best_thresh

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples or len(np.unique(y)) == 1:
            vals, counts = np.unique(y, return_counts=True)
            return {"leaf": True, "class": vals[np.argmax(counts)]}
        feat, thresh = self._best_split(X, y)
        if feat is None:
            vals, counts = np.unique(y, return_counts=True)
            return {"leaf": True, "class": vals[np.argmax(counts)]}
        mask = X[:, feat] <= thresh
        return {
            "leaf": False, "feature": feat, "threshold": thresh,
            "left": self._build(X[mask], y[mask], depth + 1),
            "right": self._build(X[~mask], y[~mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)
        return self

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["class"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)
