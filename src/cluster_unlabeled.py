# src/dbscan_outliers_location.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import DBSCAN
import umap
import matplotlib.pyplot as plt

def main():
    # 1. Carrega o CSV (ajuste o nome se necessário)
    df = pd.read_csv('new_train_labeled.csv')
    n = len(df)
    print(f"Loaded {n} rows")

    # 2. Detecta colunas de metadata com base em substrings
    video_cols = [c for c in df.columns if 'video' in c.lower()]
    frame_cols = [c for c in df.columns if any(k in c.lower() for k in ('frame','time','timestamp'))]
    print("Video columns:", video_cols)
    print("Frame/time columns:", frame_cols)

    # 3. Constrói DataFrame de metadata
    meta = pd.DataFrame({
        'original_index': df.index
    })
    for c in video_cols + frame_cols:
        meta[c] = df[c]

    # 4. Monta matriz de features (tudo que não for metadata)
    drop_cols = video_cols + frame_cols
    X = df.drop(columns=drop_cols, errors='ignore')

    # 5. Remove colunas de label caso existam
    for label in ('predicate','subject_type'):
        if label in X.columns:
            X = X.drop(columns=[label])

    # 6. Pré-processamento: MinMax + OneHot
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    print("Numeric features:", num_cols)
    print("Categorical features:", cat_cols)

    pre = ColumnTransformer([
        ('num', MinMaxScaler(), num_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
    ], remainder='drop')

    X_proc = pre.fit_transform(X)
    print("Features preprocessed:", X_proc.shape)

    # 7. DBSCAN para detecção de outliers
    db = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
    labels = db.fit_predict(X_proc)
    meta['cluster'] = labels
    meta['is_outlier'] = labels == -1

    # 8. Salva CSV com “endereço” dos outliers
    out = meta[meta['is_outlier']].copy()
    out.to_csv('outliers_location.csv', index=False)
    print(f"Saved {len(out)} outliers to outliers_location.csv")

    # 9. UMAP + plot de clusters
    um = umap.UMAP(n_components=2, random_state=42)
    emb = um.fit_transform(X_proc)

    plt.figure(figsize=(8,6))
    mask = labels != -1
    plt.scatter(emb[mask,0], emb[mask,1], c=labels[mask], cmap='Spectral', s=5, alpha=0.6)
    plt.scatter(emb[~mask,0], emb[~mask,1], c='black', s=10, label='outliers')
    plt.legend(markerscale=2)
    plt.title('DBSCAN Clusters + Outliers (UMAP 2D)')
    plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig('clusters_with_outliers.png', dpi=150)
    plt.show()
    print("Saved cluster plot to clusters_with_outliers.png")

if __name__ == '__main__':
    main()
