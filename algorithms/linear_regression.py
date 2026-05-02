import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n, m = X.shape
        self.weights = np.zeros(m)
        self.bias = 0
        for _ in range(self.epochs):
            y_pred = X @ self.weights + self.bias
            dw = (1/n) * (X.T @ (y_pred - y))
            db = (1/n) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
