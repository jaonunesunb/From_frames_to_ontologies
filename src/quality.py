from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import pandas as pd

df = pd.read_csv("data/mining_results.csv")
X = df[["avg_speed","dir_x","dir_y","duration"]]
score = silhouette_score(X, df["cluster"])
print(f"Silhouette score: {score:.3f}")


dbi = davies_bouldin_score(X, df["cluster"])
print(f"Davies–Bouldin index: {dbi:.3f}")

counts = df["cluster"].value_counts()
print(counts)