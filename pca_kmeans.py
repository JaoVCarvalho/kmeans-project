import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import manual_kmeans as manual_kmeans


def plot_clusters(X, labels, centroids, n_components, identification_name, y=None, target_names=None):
    plt.figure(figsize=(8, 6))

    if n_components == 2:
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
    elif n_components == 1:
        plt.scatter(X, np.zeros_like(X), c=labels, cmap='viridis', s=50, alpha=0.7)
        plt.scatter(centroids, np.zeros_like(centroids), c='red', s=200, marker='X', label='Centroids')
        plt.xlabel('Component 1')
        plt.yticks([])

    if y is not None and target_names is not None:
        for i, target_name in enumerate(target_names):
            plt.scatter([], [], color=plt.cm.viridis(i / len(target_names)), label=target_name)

    plt.title(f'Clustering with {n_components} PCA Components ({identification_name})')
    plt.legend()
    plt.show()

def run(identifier, n_components, k, data, random_state):
    X = data.data
    y = data.target
    target_names = data.target_names

    pca = PCA(n_components)

    if identifier == 1:
        centroids, labels = manual_kmeans.kmeans(pca.fit_transform(X), k, seed=random_state)
        plot_clusters(pca.fit_transform(X), labels, centroids, n_components,"Manual", y, target_names)

    if identifier == 2:
        kmeans = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
        kmeans.fit(pca.fit_transform(X))
        plot_clusters(pca.fit_transform(X), kmeans.labels_, kmeans.cluster_centers_, n_components,"Library", y, target_names)

