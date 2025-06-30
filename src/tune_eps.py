import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def load_embeddings(csv_path='features_unlabeled.csv'):
    df = pd.read_csv(csv_path)
    if 'video' in df.columns:
        df = df.drop(columns=['video'])
    return df.values

def plot_k_distances(X, k=5, output='k_distance.png'):
    """
    Calcula a distância até o k-ésimo vizinho para cada ponto
    e plota em ordem decrescente. O “joelho” sugere um eps.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    # Distância k-ésima para cada ponto
    k_dist = np.sort(distances[:, k-1])[::-1]
    plt.figure(figsize=(8,6))
    plt.plot(k_dist)
    plt.ylabel(f"Distância ao {k}-ésimo vizinho")
    plt.xlabel("Pontos ordenados decrescentemente")
    plt.title(f"Curva de k-Distance (k={k})")
    plt.tight_layout()
    plt.savefig(output)
    print(f"Gráfico salvo em '{output}'")
    plt.show()

def main():
    X = load_embeddings()
    plot_k_distances(X, k=5)

if __name__ == "__main__":
    main()
