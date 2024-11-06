import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# replace=false -> sem repetição
# X.shape[0] -> representa as linhas da matrix
def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]


def kmeans(X, k, max_iters=100):

    centroids = initialize_centroids(X, k)

    for _ in range(max_iters):

        # clusters com k sublinhas
        clusters = [[] for _ in range(k)]
        for point in X:

            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            # Retorna o índice do centroide mais próximo
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(point)


        old_centroids = centroids.copy()
        centroids = np.array([np.mean(cluster, axis=0) if cluster else old_centroids[i]
                              for i, cluster in enumerate(clusters)])

        if np.all(old_centroids == centroids):
            break

    return centroids, clusters

iris = load_iris()
X = iris.data

n_init = 10
best_sil_score = -1
best_centroids = None
best_labels = None

for i in range(n_init):
    centroids, clusters = kmeans(X, k=3)

    labels = []
    for i, cluster in enumerate(clusters):
        for _ in cluster:
            labels.append(i)

    sil_score = silhouette_score(X, labels)

    if sil_score > best_sil_score:
        best_sil_score = sil_score
        best_centroids = centroids
        best_labels = labels

print(f"Melhor Silhouette Score: {best_sil_score}")



