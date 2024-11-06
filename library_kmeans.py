from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.datasets import load_iris


data = load_iris()
X = data.data

for k in [3, 5]:
    print(f"Clusterização para k = {k}")

    # KMeans do sklearn com inicialização aleatória
    kmeans = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)  # n_init=10 para mais tentativas
    kmeans.fit(X)  # Ajustar o modelo aos dados

    # Obter os rótulos dos clusters (indices dos clusters atribuídos a cada ponto)
    labels = kmeans.labels_

    # Calcular o Silhouette Score
    sil_score = silhouette_score(X, labels)

    print(f"Silhouette Score: {sil_score}")
