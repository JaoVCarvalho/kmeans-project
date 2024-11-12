import tracemalloc
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris

tracemalloc.start()  # Iniciar a captura de memória

data = load_iris()
X = data.data

total_execution_time = 0
best_score = -1
best_k = None

snapshot_before = tracemalloc.take_snapshot()  # Medir o uso de memória antes da execução

for k in [3, 5]:
    start_time = time.time()
    # K-means com a biblioteca sklearn
    kmeans = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
    kmeans.fit(X)  # Ajustar o modelo aos dados
    labels = kmeans.labels_
    sil_score = silhouette_score(X, labels)
    # Captura o tempo de execução total
    end_time = time.time()  # Captura o tempo final
    execution_time = end_time - start_time  # Tempo total de execução em segundos
    total_execution_time += execution_time

    print(f"Melhor Silhouette Score para k={k}: {sil_score}")

    if sil_score > best_score:
        best_score = sil_score
        best_k = k

snapshot_after = tracemalloc.take_snapshot()  # Medir o uso de memória após a execução

# Comparar os dois snapshots para calcular a memória usada
diff = snapshot_after.compare_to(snapshot_before, 'lineno')
memory_used = sum(stat.size for stat in diff) / 1024 / 1024  # Calcular a memória total utilizada em MB

# Exibir os resultados
print(f"\nMemória total usada pelo algoritmo (em MB): {memory_used:.6f} MB")
print(f"\nO melhor Silhouette Score é {best_score} para k={best_k}.")
print(f"\nTempo total de execução (em segundos): {execution_time:.6f} segundos")
