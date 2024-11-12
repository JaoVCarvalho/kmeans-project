from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris
import tracemalloc
import time

data = load_iris()
X = data.data

best_score = -1
best_k = None
total_execution_time = 0
tracemalloc.start()
snapshot_before = tracemalloc.take_snapshot()

for k in [3, 5]:
    start_time = time.time()
    kmeans = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    sil_score = silhouette_score(X, labels)
    end_time = time.time()
    execution_time = end_time - start_time
    total_execution_time += execution_time

    print(f"O Melhor Silhouette Score para k={k}: {sil_score}")

    if sil_score > best_score:
        best_score = sil_score
        best_k = k

snapshot_after = tracemalloc.take_snapshot()
diff = snapshot_after.compare_to(snapshot_before, 'lineno')
memory_used = sum(stat.size for stat in diff) / 1024 / 1024

print(f"O melhor Silhouette Score é {best_score} para k={best_k}.")
print(f"\nMemória total utilizando a biblioteca Sklearn (em MB): {memory_used:.6f} MB")
print(f"Tempo total de execução utilizando a biblioteca Sklearn (em segundos): {execution_time:.6f} segundos")
