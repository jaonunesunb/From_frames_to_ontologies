#!/usr/bin/env python3
# cluster_hdbscan.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import hdbscan

def main():
    # 1. Carrega embeddings
    df = pd.read_csv('features_unlabeled.csv')
    X = df.drop(columns=['video']).values if 'video' in df.columns else df.values

    # 2. Padroniza
    Xs = StandardScaler().fit_transform(X)

    # 3. HDBSCAN: ajuste min_cluster_size conforme o que espera de tamanho mínimo de cluster
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
    labels = clusterer.fit_predict(Xs)
    df['hdb_cluster'] = labels

    # 4. Estatísticas
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise   = (labels == -1).sum()
    print(f"HDBSCAN → {n_clusters} clusters, {n_noise} outliers")

    # 5. Salva
    df.to_csv('features_unlabeled_hdbscan.csv', index=False)
    print("Salvo em 'features_unlabeled_hdbscan.csv'")

if __name__ == "__main__":
    main()
