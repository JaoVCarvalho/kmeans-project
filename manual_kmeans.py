import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
import tracemalloc
import time
import pca_kmeans as pca_kmeans

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def initialize_centroids(X, k, seed=None):
    np.random.seed(seed)
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def kmeans(X, k, max_iters=100, tol=1e-4, seed=None):
    centroids = initialize_centroids(X, k, seed)

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                                  for i in range(k)])

        if np.all(np.abs(new_centroids - centroids) < tol):
            break

        centroids = new_centroids

    return centroids, labels

def print_default_message():
    print("=========================================================="
          + f"\n           K-means Clustering - Manual (Hardcore)"
          + "\n==========================================================\n"
          + "\nK-means clustering implemented manually, initializing centroids\n"
            "assigning points to the nearest centroids, and recalculating centroids\n"
            "iteratively until convergence. The Silhouette Score is calculated\n"
            "to evaluate clustering quality, without using external libraries for clustering.\n\n")

def run(random_state = 42, n_init = 10):

    print_default_message()

    iris = load_iris()
    X = iris.data

    best_sil_scores = {}
    total_execution_time = 0
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    for k in [3, 5]:
        start_time = time.time()
        best_sil_score = -1
        best_centroids = None
        best_labels = None

        for _ in range(n_init):
            centroids, labels = kmeans(X, k, seed=random_state)
            sil_score = silhouette_score(X, labels)

            if sil_score > best_sil_score:
                best_sil_score = sil_score
                best_centroids = centroids
                best_labels = labels

        end_time = time.time()
        execution_time = end_time - start_time
        total_execution_time += execution_time

        best_sil_scores[k] = best_sil_score
        print(f"Best Silhouette Score for k={k}: {best_sil_score}")

    best_k, best_score = max(best_sil_scores.items(), key=lambda item: item[1])

    snapshot_after = tracemalloc.take_snapshot()
    diff = snapshot_after.compare_to(snapshot_before, 'lineno')
    memory_used = sum(stat.size for stat in diff) / 1024 / 1024  # Converter para MB

    print(f"\nBest Silhouette Score is {best_score} for k={best_k}.")
    print(f"\nTotal memory used by the algorithm (in MB) {memory_used:.6f} MB")
    print(f"Total execution time using the algorithm (in seconds): {total_execution_time:.6f} seconds \n")

    pca_kmeans.run(1,1,best_k,iris,42)
    pca_kmeans.run(1, 2, best_k, iris, 42)

    print(f"==========================================================\n\n")

if __name__ == "__main__":
    run(42, 10)