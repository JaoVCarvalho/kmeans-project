from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris
import tracemalloc
import time
import pca_kmeans as pca_kmeans

def print_default_message():
    print("=========================================================="
          + f"\n           K-means Clustering - Library (Sklearn)"
          + "\n==========================================================\n"
          + "\nK-means clustering using libraries like scikit-learn, \n"
            "simplifying implementation and improving efficiency with \n"
            "built-in functions for centroid initialization, point assignment, \n"
            "and iterative updates, along with Silhouette Score calculation \n"
            "for cluster quality evaluation \n \n")

def run(random_state = 42, n_init = 10):

    print_default_message()

    iris = load_iris()
    X = iris.data

    best_score = -1
    best_k = None
    total_execution_time = 0
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    for k in [3, 5]:
        start_time = time.time()
        kmeans = KMeans(n_clusters=k, init='random', n_init=n_init, random_state=random_state)
        kmeans.fit(X)
        labels = kmeans.labels_
        sil_score = silhouette_score(X, labels)
        end_time = time.time()
        execution_time = end_time - start_time
        total_execution_time += execution_time

        print(f"Best Silhouette Score for k={k}: {sil_score}")

        if sil_score > best_score:
            best_score = sil_score
            best_k = k

    snapshot_after = tracemalloc.take_snapshot()
    diff = snapshot_after.compare_to(snapshot_before, 'lineno')
    memory_used = sum(stat.size for stat in diff) / 1024 / 1024

    print(f"\nBest Silhouette Score is {best_score} for k={best_k}.")

    print(f"\nTotal memory used by Sklearn library (in MB): {memory_used:.6f} MB")
    print(f"Total execution time using the Sklearn library (in seconds): {execution_time:.6f} seconds")

    pca_kmeans.run(2, 1, best_k, iris, 42)
    pca_kmeans.run(2, 2, best_k, iris, 42)

    print(f"==========================================================\n\n")

if __name__ == "__main__":
    run(42, 10)