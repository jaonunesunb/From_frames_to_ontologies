#!/usr/bin/env python3
# extract_anomalies.py

import pandas as pd
from pathlib import Path

def main():
    df = pd.read_csv('features_unlabeled_clusters_eps5.csv')
    anoms = df[df['cluster'] == -1]['video'].tolist()
    print(f"Total de anomalias: {len(anoms)}")
    with open('anomaly_videos.txt', 'w', encoding='utf-8') as f:
        for v in anoms:
            f.write(v + '\n')
    print("Salvo 'anomaly_videos.txt'.")

if __name__ == "__main__":
    main()
