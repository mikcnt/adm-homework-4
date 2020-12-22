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
        centroids = []
        for k in range(self.K):
            cluster_k = self.X[self.labels_ == k]
            if cluster_k.shape[0] == 0:
                centroids.append(self.cluster_centers_[k])
            else:
                centroids.append(cluster_k.mean(axis=0))
        self.cluster_centers_ = np.array(centroids)


def elbow_method(data, min_cluster_n=2, max_cluster_n=10, threshold=0.965):
    cluster_values = range(min_cluster_n, max_cluster_n + 1)
    sum_squares = []
    for k in cluster_values:
        model = KMeansClustering(n_clusters=k)
        model.fit(data)
        sum_squares.append(model.inertia_)
    
    sum_squares = np.array(sum_squares)
    slope = np.append(sum_squares[1:], 0) / sum_squares
    
    stopping_points = np.where(slope > threshold)[0]
    
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
    
def plot_wordclouds(data, clusters, cluster_col = 'Cluster'):
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
        cl_n = clusters[0]
        for i in range(sub_rows):
            for j in range(sub_cols):
                if cl_n - clusters[0] < n_clusters:
                    axs[i, j].imshow(word_cloud(data, cl_n, cluster_col=cluster_col))
                    axs[i, j].set_title('Cluster {}'.format(cl_n))
                axs[i, j].axis('off')
                cl_n += 1
    
    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    for ax in axs.flat:
        ax.label_outer()
        
def reviews_per_cluster(data):
    display(data.groupby('Cluster').count()['Text'].reset_index().rename(columns={'Text': 'Reviews Number'}))
    
def plot_distributions(data, n_clusters=16):
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