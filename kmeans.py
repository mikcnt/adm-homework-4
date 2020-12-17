import numpy as np
import matplotlib.pyplot as plt

def dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:
    def __init__(self, K=5, max_iters=100, plot_figure=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_figure = plot_figure
        
    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        centroids = self._initialize_centroids()
        for _ in range(self.max_iters):
            clusters = self._create_clusters(centroids)
            
            previous_centroids = centroids
            centroids = self._new_centroids(clusters)
            
            diff = centroids - previous_centroids
            
            if not diff.any():
                break
        
        y_pred = self._predict(clusters)
        
        if self.plot_figure:
            self.plot_fig(clusters, centroids)

        return y_pred
        
        
    def _initialize_centroids(self):
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        centroids = [self.X[idx] for idx in random_sample_idxs]
        return centroids
    
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            distances = [dist(sample, point) for point in centroids]
            closest_index = np.argmin(distances)
            clusters[closest_index].append(idx)
        return clusters
    
    def _new_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(X[cluster], axis=0)
            centroids[idx] = new_centroid
        return centroids
    
    def _predict(self, clusters):
        y_pred = np.zeros(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
                
        return y_pred
    
    def plot_fig(self, clusters, centroids):
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Plot clusters
        for cluster in clusters:
            cluster_points = self.X[cluster].T
            ax.scatter(*cluster_points)
            
        for centroid in centroids:
            ax.scatter(*centroid, marker='x', color='black', linewidth=2)
        plt.show()