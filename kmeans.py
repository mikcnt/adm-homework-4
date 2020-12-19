import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def euc_dist(X, Y):
    return np.sqrt(np.sum((X[:, None] - Y) ** 2, axis=2))


def sed_distance(X, Y):
    return sum(np.min(np.sum((X[:, None] - Y) ** 2, axis=2), axis=1))


class KMeans:
    def __init__(self, n_clusters, max_iters=100, plot_figure=False):
        self.K = n_clusters
        self.max_iters = max_iters
        self.plot_figure = plot_figure

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        self.cluster_centers_ = self._initialize_centroids()
        for _ in range(self.max_iters):
            self._create_clusters()

            previous_centroids = self.cluster_centers_
            self.cluster_centers_ = self._new_centroids()
            self.inertia_ = sed_distance(self.X, self.cluster_centers_)

            diff = self.cluster_centers_ - previous_centroids

            if not diff.any():
                break

        if self.plot_figure:
            self.plot_fig()
        return self

    def _initialize_centroids(self):
        np.random.seed(42)
        random_sample_idxs = np.random.choice(
            self.n_samples, self.K, replace=False)
        centroids = self.X[random_sample_idxs]
        return centroids

    def _create_clusters(self):
        self.labels_ = euc_dist(self.X, self.cluster_centers_).argmin(axis=1)

    def _new_centroids(self):
        centroids = np.array([self.X[self.labels_ == k].mean(axis=0) for k in range(self.K)])
        return centroids

    def plot_fig(self):
        assert(self.n_features == 2), 'Cannot plot kmeans if there are more than 2 dimensions.'
        _, ax = plt.subplots(figsize=(16, 12))

        for i in range(self.K):
            cluster_points = np.array([self.X[self.labels_ == i]]).T
            ax.scatter(*cluster_points)
