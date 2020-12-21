import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy import linalg as LA

not_fitted_error = "This KMeans instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
wrong_data_error = "Call 'predict' on the same data used to fit the KMeans model."

np.random.seed(42)

def elbow_method(X, K=10):
    sum_squares = []
    cluster_values = range(1, K + 1)
    for k in cluster_values:
        model = KMeansClustering(n_clusters=k)
        model.fit(X)
        sum_squares.append(model.inertia_)
        print('Number of clusters = {}\t'
              'Number of iterations = {}'
              .format(k, model.n_iter_))
    
    sum_squares = np.array(sum_squares)
    sum_squares_normalized = sum_squares / sum_squares.max()
    
    plt.figure(figsize=(12, 6))
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    plt.xticks(cluster_values)
    plt.plot(cluster_values, sum_squares_normalized, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Normalized sum of squares')
    plt.title('Elbow method')
    plt.show()
    


def dist(X, Y):
    return np.sqrt(np.sum((X[:, None] - Y) ** 2, axis=2))


def sed_distance(X, Y):
    return sum(np.min(np.sum((X[:, None] - Y) ** 2, axis=2), axis=1))
class KMeansClustering:
    def __init__(self, n_clusters, max_iter=150, plot_figure=False):
        self.K = n_clusters
        self.max_iters = max_iter
        self.plot_figure = plot_figure
        self.inertia_ = 0

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        self._initialize_centroids()
        for self.n_iter_ in range(self.max_iters):
            self._create_clusters()
            previous_centroids = self.cluster_centers_
            self._new_centroids()
            self.inertia_ = sed_distance(self.X, self.cluster_centers_)
            # Strong convergence: no differences between centroids
            diff = self.cluster_centers_ - previous_centroids
            if not diff.any():
                break
            # Weak convergence: close centroids between two iterations
            if LA.norm(diff) < 1e-04:
                break
        self.n_iter_ += 1
        return self
    
    def predict(self, X):
        if not hasattr(self, 'labels_'):
            raise AttributeError(not_fitted_error)
        if (self.X - X).any():
            raise ValueError(wrong_data_error)
        return self.labels_
    
    def fit_predict(self, X):
        return self.fit(X).labels_

    def _initialize_centroids(self):
        np.random.seed(42)
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        centroids = self.X[random_sample_idxs]
        self.cluster_centers_ = centroids

    def _create_clusters(self):
        self.labels_ = dist(self.X, self.cluster_centers_).argmin(axis=1)

    def _new_centroids(self):
        centroids = np.array([self.X[self.labels_ == k].mean(axis=0) for k in range(self.K)])
        self.cluster_centers_ = centroids