import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

best_k = 3
kmeans = KMeans(n_clusters=best_k, init='random', n_init=10, random_state=42)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Função para plotar clusters e centróides
def plot_clusters(X, labels, centroids, n_components, y=None, target_names=None):
    plt.figure(figsize=(8, 6))

    if n_components == 2:
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
        plt.xlabel('Componente 1')
        plt.ylabel('Componente 2')
    elif n_components == 1:
        plt.scatter(X, np.zeros_like(X), c=labels, cmap='viridis', s=50, alpha=0.7)
        plt.scatter(centroids, np.zeros_like(centroids), c='red', s=200, marker='X', label='Centroids')
        plt.xlabel('Componente 1')
        plt.yticks([])  # Não mostrar o eixo y

    if y is not None and target_names is not None:
        for i, target_name in enumerate(target_names):
            plt.scatter([], [], color=plt.cm.viridis(i / len(target_names)), label=target_name)

    plt.title(f'Clusterização com {n_components} Componentes PCA')
    plt.legend()
    plt.show()

pca_1 = PCA(n_components=1)
X_pca_1 = pca_1.fit_transform(X)
kmeans_pca_1 = KMeans(n_clusters=best_k, init='random', n_init=10, random_state=42)
kmeans_pca_1.fit(X_pca_1)
labels_1 = kmeans_pca_1.labels_
centroids_1 = kmeans_pca_1.cluster_centers_

pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X)
kmeans_pca_2 = KMeans(n_clusters=best_k, init='random', n_init=10, random_state=42)
kmeans_pca_2.fit(X_pca_2)
labels_2 = kmeans_pca_2.labels_
centroids_2 = kmeans_pca_2.cluster_centers_

plot_clusters(X_pca_1, labels_1, centroids_1, 1, y, target_names)
plot_clusters(X_pca_2, labels_2, centroids_2, 2, y, target_names)