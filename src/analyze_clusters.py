import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def load_data(csv_path='features_unlabeled_clusters.csv'):
    df = pd.read_csv(csv_path)
    if 'cluster' not in df.columns:
        raise ValueError("O CSV não contém a coluna 'cluster'")
    return df

def cluster_counts(df):
    counts = df['cluster'].value_counts().sort_index()
    print("Contagem de vídeos por cluster:")
    print(counts.to_string())
    return counts

def plot_pca(df, output_path='clusters_pca.png'):
    X = df.drop(columns=['video','cluster']).values
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    df['PC1'] = X_pca[:,0]
    df['PC2'] = X_pca[:,1]

    plt.figure(figsize=(8,6))
    for lbl in sorted(df['cluster'].unique()):
        mask = df['cluster'] == lbl
        plt.scatter(df.loc[mask,'PC1'], df.loc[mask,'PC2'],
                    label=f'Cluster {lbl}', alpha=0.6, s=30)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clusters DBSCAN em PCA 2D')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Gráfico PCA salvo em '{output_path}'")

def inspect_examples(df, n=5):
    print("\nExemplos por cluster:")
    for lbl in sorted(df['cluster'].unique()):
        subset = df[df['cluster'] == lbl]['video'].head(n).tolist()
        print(f"Cluster {lbl} (n={len(df[df.cluster==lbl])}): {subset}")

def compute_silhouette(df):
    X = df.drop(columns=['video','cluster']).values
    labels = df['cluster'].values
    # silhouette_score não aceita -1, então ignora ruído
    mask = labels != -1
    if len(set(labels[mask])) < 2:
        print("\nSilhouette Score: não há clusters suficientes (excluindo ruído).")
        return
    score = silhouette_score(X[mask], labels[mask])
    print(f"\nSilhouette Score (excluindo -1): {score:.4f}")

def main():
    df = load_data()
    cluster_counts(df)
    plot_pca(df)
    inspect_examples(df, n=5)
    compute_silhouette(df)

if __name__ == "__main__":
    main()
