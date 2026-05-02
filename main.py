#!/usr/bin/env python3
"""Demo: Run all ML algorithms on synthetic data."""
import numpy as np
from algorithms.linear_regression import LinearRegression
from algorithms.logistic_regression import LogisticRegression
from algorithms.kmeans import KMeans
from algorithms.decision_tree import DecisionTree
from algorithms.knn import KNN

np.random.seed(42)

def demo_linear():
    print("=== Linear Regression ===")
    X = np.random.randn(200, 1) * 2
    y = 3 * X.squeeze() + 7 + np.random.randn(200) * 0.5
    model = LinearRegression(lr=0.05, epochs=500).fit(X, y)
    print(f"R² Score: {model.score(X, y):.4f}")
    print(f"Weights: {model.weights[0]:.3f}, Bias: {model.bias:.3f} (expected ~3, ~7)\n")

def demo_logistic():
    print("=== Logistic Regression ===")
    X = np.vstack([np.random.randn(100, 2) + [2, 2], np.random.randn(100, 2) + [-2, -2]])
    y = np.array([1]*100 + [0]*100)
    model = LogisticRegression(lr=0.1, epochs=300).fit(X, y)
    print(f"Accuracy: {model.accuracy(X, y)*100:.1f}%\n")

def demo_kmeans():
    print("=== K-Means Clustering ===")
    X = np.vstack([np.random.randn(50, 2) + c for c in [[0,0],[5,5],[10,0]]])
    model = KMeans(k=3).fit(X)
    print(f"Inertia: {model.inertia(X):.2f}")
    print(f"Centroids:\n{model.centroids}\n")

def demo_tree():
    print("=== Decision Tree ===")
    X = np.vstack([np.random.randn(100, 2) + [2, 2], np.random.randn(100, 2) + [-2, -2]])
    y = np.array([1]*100 + [0]*100)
    model = DecisionTree(max_depth=5).fit(X, y)
    print(f"Accuracy: {model.accuracy(X, y)*100:.1f}%\n")

def demo_knn():
    print("=== K-Nearest Neighbors ===")
    X = np.vstack([np.random.randn(100, 2) + [3, 3], np.random.randn(100, 2) + [-3, -3]])
    y = np.array([1]*100 + [0]*100)
    model = KNN(k=5).fit(X, y)
    print(f"Accuracy: {model.accuracy(X, y)*100:.1f}%\n")

if __name__ == "__main__":
    demo_linear()
    demo_logistic()
    demo_kmeans()
    demo_tree()
    demo_knn()
