import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from numpy import linalg as LA

not_fitted_error = "This KMeans instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
wrong_data_error = "Call 'predict' on the same data used to fit the KMeans model."

np.random.seed(42)

def dist(X, Y):
    """Euclidean distance."""
    return np.sqrt(np.sum((X[:, None] - Y) ** 2, axis=2))

def sed_distance(X, Y):
    """Sum of square distances."""
    return sum(np.min(np.sum((X[:, None] - Y) ** 2, axis=2), axis=1))

class KMeansClustering:
    """Custom K-Means clustering.

    Args:
        n_clusters (int): Number of clusters to create.
        max_iter (int, optional): Number of iterations to try before quitting the fit method. Defaults to 150.
    """
    def __init__(self, n_clusters, max_iter=150, plot_figure=False):
        self.K = n_clusters
        self.max_iters = max_iter
        self.inertia_ = 0

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        # First initialization is random
        self._initialize_centroids()
        for self.n_iter_ in range(self.max_iters):
            # Compute centroids from previous centroids
            self._create_clusters()
            # Keep old centroids to check for convergence
            previous_centroids = self.cluster_centers_
            self._new_centroids()
            # Compute sum of square distances for error (useful when using elbow method)
            self.inertia_ = sed_distance(self.X, self.cluster_centers_)
            # Strong convergence: no differences between centroids
            diff = self.cluster_centers_ - previous_centroids
            if not diff.any():
                break
            # Weak convergence: close distance between centroids of two following iterations
            if LA.norm(diff) < 1e-04:
                break
            self.n_iter_ += 1
        # Rename centroids based on their population
        self._order_labels()
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
        centroids = []
        for k in range(self.K):
            cluster_k = self.X[self.labels_ == k]
            if cluster_k.shape[0] == 0:
                centroids.append(self.cluster_centers_[k])
            else:
                centroids.append(cluster_k.mean(axis=0))
        self.cluster_centers_ = np.array(centroids)
        
    def _order_labels(self):
        cluster_sizes = []
        for i in range(self.K):
            cluster_sizes.append(self.labels_[self.labels_ == i].size)

        ordered_sizes = np.argsort(-np.array(cluster_sizes))

        ordered_labels = np.zeros_like(self.labels_)
        for i, label in enumerate(ordered_sizes):
            ordered_labels[self.labels_ == label] = i
            
        self.labels_ = ordered_labels


def elbow_method(data, min_cluster_n=2, max_cluster_n=10, threshold=0.965):
    """Plot the sum of square distances for the K-Means algorithm as the number of clusters increase.

    Args:
        data (np.array): Data to use for the computation of the error.
        min_cluster_n (int, optional): Minimum number of clusters to consider. Defaults to 2.
        max_cluster_n (int, optional): Maximum number of clusters to consider. Defaults to 10.
        threshold (float, optional): Threshold used to find sweet spot. Defaults to 0.965.
    """
    cluster_values = range(min_cluster_n, max_cluster_n + 1)
    sum_squares = []
    # Compute kmeans for each number of clusters
    for k in cluster_values:
        model = KMeansClustering(n_clusters=k)
        model.fit(data)
        sum_squares.append(model.inertia_)
    
    # Compute the slopes comparing the values between one number of clusters and the following one
    sum_squares = np.array(sum_squares)
    slope = np.append(sum_squares[1:], 0) / sum_squares
    
    # Sweet spot
    stopping_points = np.where(slope > threshold)[0]
    
    # It is possible that there is no sweet spot in the range we consider
    if stopping_points.size != 0:
        stopping_point = stopping_points[0] + cluster_values[0]
    
    plt.figure(figsize=(12, 6))
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    plt.xticks(cluster_values)
    plt.plot(cluster_values, sum_squares, 'o-')
    if stopping_points.size != 0:
        plt.axvline(x=stopping_point, color='black', linestyle='--')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squares')
    plt.title('Elbow method')
    plt.show()        

def word_cloud(data, cluster_n, min_df=0.01, max_df=1.0, max_words=80, plot_dim=(800, 800), cluster_col = 'Cluster'):
    """Helper function to compute wordcloud for a certain cluster."""
    width, height = plot_dim
    cluster_data = data[data[cluster_col] == cluster_n]
    vect = TfidfVectorizer(min_df=min_df, max_df=max_df)
    vecs = vect.fit_transform(cluster_data['Text'])
    feature_names = vect.get_feature_names()
    dense = vecs.todense()
    lst1 = dense.tolist()
    df = pd.DataFrame(lst1, columns=feature_names)
    
    return WordCloud(width=width, height=height,
                     max_words=max_words,
                     background_color ='white',
                     min_font_size = 10).generate_from_frequencies(df.T.sum(axis=1))
    
def plot_wordclouds(data, clusters, cluster_col='Cluster'):
    """Plot multiple wordclouds all at once.

    Args:
        data (pd.DataFrame): Data used to retrieve tfidf values for the wordclouds.
        clusters (list): List of clusters for which we want to plot the wordcloud.
        cluster_col (str, optional): Column descriptor for the clustering algorithm 
                                     ('Cluster_sklearn' for the sklearn algorithm). Defaults to 'Cluster'.
    """
    n_clusters = len(clusters)
    sub_rows = n_clusters // 3 if n_clusters % 3 == 0 else n_clusters // 3 + 1
    sub_cols = 3 if n_clusters >= 3 else n_clusters
    fig_w = 22
    fig_h = 7 * sub_rows
    
    _, axs = plt.subplots(sub_rows, sub_cols, figsize=(fig_w, fig_h))
    if sub_rows == 1:
        for i, cl_n in enumerate(clusters):
            axs[i].imshow(word_cloud(data, cl_n, cluster_col=cluster_col))
            axs[i].set_title('Cluster {}'.format(cl_n))
            axs[i].axis('off')
    else:
        cont = 0
        for i in range(sub_rows):
            for j in range(sub_cols):
                if cont < n_clusters:
                    axs[i, j].imshow(word_cloud(data, clusters[cont], cluster_col=cluster_col))
                    axs[i, j].set_title('Cluster {}'.format(clusters[cont]))
                axs[i, j].axis('off')
                cont += 1
    
    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    for ax in axs.flat:
        ax.label_outer()
        
def wordcloud_comparison(data, cluster):
    """Compare wordclouds generated from custom and sklearn K-Means algorihtms."""
    sk_clusters_ordered = (- data.groupby('Cluster_sklearn').count()['Time']).argsort().tolist()
    sk_cluster = sk_clusters_ordered[cluster]
    _, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].imshow(word_cloud(data, cluster, cluster_col='Cluster'))
    axs[0].set_title('Custom KMeans - Cluster {}'.format(cluster), fontsize=14)
    axs[0].axis('off')
    axs[1].imshow(word_cloud(data, sk_cluster, cluster_col='Cluster_sklearn'))
    axs[1].set_title('Sklearn KMeans - Cluster {}'.format(sk_cluster), fontsize=14)
    axs[1].axis('off')
    
    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    for ax in axs.flat:
        ax.label_outer()
        
def reviews_per_cluster(data, cluster_col='Cluster'):
    return data.groupby(cluster_col).count()['Text'].reset_index().rename(columns={'Text': 'Reviews Number'})
    
def plot_distributions(data, n_clusters=16):
    """Plot distributions for the score values in each cluster."""
    bins = np.arange(7) - 0.5
    _, ax = plt.subplots(4, 4, figsize=(20, 20))
    plt.setp(ax, xticks=range(1, 6), xlim=[0, 6])
    ax = ax.reshape(16)
    for c in range(n_clusters):
        cluster_score = data[data['Cluster'] == c]['Score']
        ax[c].hist(cluster_score, bins=bins, alpha=0.8, edgecolor='black')
    plt.show()
    
def uniqueusers_per_cluster(data):
    display(data.groupby(['Cluster'])['UserId'].nunique().reset_index())

def compare_clusters(data, cluster):
    """Compare two clusters (custom and sklearn) by their intersection."""
    revs_custom = reviews_per_cluster(data)
    revs_sklearn = reviews_per_cluster(data, cluster_col='Cluster_sklearn')

    n_custom = np.array(revs_custom['Reviews Number'])
    n_sklearn = np.array(revs_sklearn['Reviews Number'])

    sk_cluster = np.argmin(np.abs(n_custom[:, None] - n_sklearn), axis=1)[cluster]
    
    num = data[(data['Cluster'] == cluster) & (data['Cluster_sklearn'] == sk_cluster)].shape[0]
    den_1 = data[data['Cluster'] == cluster].shape[0]
    den_2 = data[data['Cluster_sklearn'] == sk_cluster].shape[0]
    prob_1 = num / den_1
    prob_2 = num / den_2
    
    return sk_cluster, prob_1, prob_2