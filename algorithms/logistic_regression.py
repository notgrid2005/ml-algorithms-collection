import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n, m = X.shape
        self.weights = np.zeros(m)
        self.bias = 0
        for _ in range(self.epochs):
            z = X @ self.weights + self.bias
            a = self._sigmoid(z)
            dw = (1/n) * (X.T @ (a - y))
            db = (1/n) * np.sum(a - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)
