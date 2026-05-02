import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def fit(self, X):
        idx = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[idx].copy()
        for _ in range(self.max_iters):
            dists = np.linalg.norm(X[:, None] - self.centroids, axis=2)
            self.labels = np.argmin(dists, axis=1)
            new_centroids = np.array([
                X[self.labels == i].mean(axis=0) if np.any(self.labels == i)
                else self.centroids[i] for i in range(self.k)
            ])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return self

    def predict(self, X):
        dists = np.linalg.norm(X[:, None] - self.centroids, axis=2)
        return np.argmin(dists, axis=1)

    def inertia(self, X):
        labels = self.predict(X)
        return sum(np.sum((X[labels == i] - self.centroids[i]) ** 2) for i in range(self.k))
