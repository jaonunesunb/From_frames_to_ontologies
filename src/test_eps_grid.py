#!/usr/bin/env python3
# test_eps_grid.py

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def load_data(csv_path='features_unlabeled.csv'):
    df = pd.read_csv(csv_path)
    vids = df['video'] if 'video' in df.columns else None
    X = df.drop(columns=['video']).values if vids is not None else df.values
    Xs = StandardScaler().fit_transform(X)
    return Xs

def test_grid(eps_values, min_samples=5):
    Xs = load_data()
    results = []
    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
        labels = db.fit_predict(Xs)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        results.append((eps, n_clusters, n_noise))
    return results

def main():
    # ajuste esta lista após olhar o k-distance !
    eps_values = np.linspace(0.5, 5.0, 10)
    res = test_grid(eps_values, min_samples=5)
    df = pd.DataFrame(res, columns=['eps', 'n_clusters', 'n_noise'])
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
