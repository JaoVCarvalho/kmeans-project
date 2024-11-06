from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.datasets import load_iris


data = load_iris()
X = data.data

for k in [3, 5]:
    print(f"Clusterização para k = {k}")


    kmeans = KMeans(n_clusters=k, init='random', n_init=10, random_state=42) 
    kmeans.fit(X)  # Ajustar o modelo aos dados

    labels = kmeans.labels_

    sil_score = silhouette_score(X, labels)

    print(f"Silhouette Score: {sil_score}")
