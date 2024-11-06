import numpy as np
from sklearn.datasets import load_iris

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


# Carrega o conjunto de dados Iris (desconsiderando o rótulo)
iris = load_iris()
X = iris.data  # Dados sem o rótulo

# Executa o K-means com k=3
k3_centroids, k3_clusters = kmeans(X, k=3)
print("Centroides finais para k=3:\n", k3_centroids)
print("Cluester k = 3: \n", k3_clusters)

# Executa o K-means com k=5
k5_centroids, k5_clusters = kmeans(X, k=5)
print("Centroides finais para k=5:\n", k5_centroids)
print("Cluester k = 5: \n", k5_clusters)


