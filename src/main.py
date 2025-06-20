import os
import pandas as pd
from dataset import list_all_clips
from video_processing import process_and_cache
from feature_extraction import extract_features
from mining import run_mining
from config import FEATURES_CSV

if __name__ == "__main__":
    # 1) Process videos and extract features
    all_rows = []
    for clip in list_all_clips():
        traj, fps = process_and_cache(clip)
        df_feat = extract_features(traj, fps)
        df_feat['video'] = os.path.basename(clip)
        all_rows.append(df_feat)
    df_all = pd.concat(all_rows, ignore_index=True)
    os.makedirs(os.path.dirname(FEATURES_CSV), exist_ok=True)
    df_all.to_csv(FEATURES_CSV, index=False)
    print(f"[main] Features saved to {FEATURES_CSV}")
    # 2) Run clustering
    df_clusters = run_mining()
    print(df_clusters['cluster'].value_counts())
