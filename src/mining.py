import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from config import FEATURES_CSV, MINING_CSV, DBSCAN_EPS, DBSCAN_MIN_SAMPLES

def run_mining():
    df = pd.read_csv(FEATURES_CSV)
    features = ['avg_speed','dir_x','dir_y','duration']
    X = StandardScaler().fit_transform(df[features])
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(X)
    df['cluster'] = labels
    df.to_csv(MINING_CSV, index=False)
    print(f"[mining] Saved clustering results to {MINING_CSV}")
    return df
