import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
import tracemalloc
import time

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def initialize_centroids(X, k, seed=None):
    np.random.seed(seed)  # Definindo a semente aleatória para garantir resultados consistentes
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def kmeans(X, k, max_iters=100, tol=1e-4, seed=None):
    centroids = initialize_centroids(X, k, seed)
    prev_centroids = centroids.copy()

    for _ in range(max_iters):
        # Passo 1: Atribuir cada ponto ao centróide mais próximo
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # Calcula todas as distâncias de uma vez
        labels = np.argmin(distances, axis=1)  # Atribui o ponto ao centróide mais próximo

        # Passo 2: Recalcular os centróides
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                                  for i in range(k)])

        # Verificar convergência: se os centróides não mudaram muito, parar
        if np.all(np.abs(new_centroids - prev_centroids) < tol):
            break

        prev_centroids = new_centroids.copy()  # Atualizar centróides anteriores

    return new_centroids, labels

iris = load_iris()
X = iris.data

best_sil_scores = {}
seed = 42
total_execution_time = 0
tracemalloc.start()
snapshot_before = tracemalloc.take_snapshot()

for k in [3, 5]:
    start_time = time.time()  # Captura o tempo inicial para o k atual
    best_sil_score = -1
    best_centroids = None
    best_labels = None

    for _ in range(10):  # Realizar múltiplas inicializações para encontrar o melhor Silhouette Score
        centroids, labels = kmeans(X, k, seed=seed)
        sil_score = silhouette_score(X, labels)

        if sil_score > best_sil_score:
            best_sil_score = sil_score
            best_centroids = centroids
            best_labels = labels

    end_time = time.time()  # Captura o tempo final para o k atual
    execution_time = end_time - start_time  # Tempo de execução para o k atual
    total_execution_time += execution_time  # Soma ao tempo total de execução

    # Armazena o melhor Silhouette Score para o k atual
    best_sil_scores[k] = best_sil_score
    print(f"O Melhor Silhouette Score para k={k}: {best_sil_score}")

# Encontrar o melhor k com base no Silhouette Score
best_k, best_score = max(best_sil_scores.items(), key=lambda item: item[1])

snapshot_after = tracemalloc.take_snapshot()
diff = snapshot_after.compare_to(snapshot_before, 'lineno')
memory_used = sum(stat.size for stat in diff) / 1024 / 1024  # Converter para MB

print(f"O melhor Silhouette Score é {best_score} para k={best_k}.")
print(f"\nMemória total usada pelo algoritmo (em MB): {memory_used:.6f} MB")
print(f"Tempo total de execução (em segundos): {total_execution_time:.6f} segundos")
