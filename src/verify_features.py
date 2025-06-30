import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_lists(videos_txt='videos_sem_anotacao.txt',
               features_csv='features_unlabeled.csv'):
    # Vídeos esperados (stems)
    expected = set(Path(line.strip()).stem
                   for line in open(videos_txt, encoding='utf-8')
                   if line.strip())
    # Carrega features
    df = pd.read_csv(features_csv)
    processed = set(Path(v).stem for v in df['video'])
    return expected, df, processed

def check_completeness(expected, processed):
    missed = expected - processed
    extra  = processed - expected
    print(f"Esperados: {len(expected)}, Processados: {len(processed)}")
    print(f"Não processados (faltantes): {len(missed)}")
    if missed:
        print(sorted(list(missed))[:10], "…")
    print(f"Processados mas não esperados: {len(extra)}")
    if extra:
        print(sorted(list(extra))[:10], "…")
    return missed, extra

def check_nans(df):
    na_counts = df.isna().sum()
    print("\nValores faltantes por coluna:")
    print(na_counts[na_counts>0].to_string() or "Nenhum NaN encontrado")
    return na_counts

def stats_overview(df):
    desc = df.drop(columns=['video']).describe().T
    print("\nEstatísticas descritivas das features:")
    print(desc[['count','mean','std','min','50%','max']])
    return desc

def plot_histograms(df, n_cols=3):
    feats = df.drop(columns=['video'])
    cols = feats.columns
    # Plotar apenas primeitas n_cols features
    for col in cols[:n_cols]:
        plt.figure(figsize=(6,4))
        plt.hist(feats[col], bins=30)
        plt.title(f"Histograma de {col}")
        plt.xlabel(col)
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.savefig(f"hist_{col}.png")
        print(f"Salvo histograma em hist_{col}.png")
    print(f"\nPlotei histogramas das primeiras {n_cols} features.")

def main():
    expected, df, processed = load_lists()
    check_completeness(expected, processed)
    check_nans(df)
    stats_overview(df)
    plot_histograms(df, n_cols=5)

if __name__ == "__main__":
    main()
