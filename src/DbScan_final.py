import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Carregar dataset
df = pd.read_csv("src/outputs/paired_train_labeled_cleaned.csv")
target_col = 'subject_type'

# Filtrar classes com pelo menos 10 instâncias
class_counts = df[target_col].value_counts()
valid_classes = class_counts[class_counts >= 10].index
df = df[df[target_col].isin(valid_classes)]

# Features e Label
X = df.drop(columns=[target_col])
y = df[target_col]

# Label Encoding
categorical_cols = X.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Normalização
numeric_cols = X.select_dtypes(include='number').columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=10)
db_labels = dbscan.fit_predict(X_pca)

# HDBSCAN
hdb = hdbscan.HDBSCAN(min_cluster_size=10)
hdb_labels = hdb.fit_predict(X_pca)

# Adicionar ao dataframe original
df_result = df.copy()
df_result['PCA1'] = X_pca[:, 0]
df_result['PCA2'] = X_pca[:, 1]
df_result['dbscan_cluster'] = db_labels
df_result['hdbscan_cluster'] = hdb_labels

# Salvar outliers
dbscan_outliers = df_result[df_result['dbscan_cluster'] == -1][['video', 'frame', 'subject_type']]
hdbscan_outliers = df_result[df_result['hdbscan_cluster'] == -1][['video', 'frame', 'subject_type']]
dbscan_outliers.to_csv("src/outputs/dbscan_outliers.csv", index=False)
hdbscan_outliers.to_csv("src/outputs/hdbscan_outliers.csv", index=False)

# Visualização DBSCAN
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='dbscan_cluster', data=df_result, palette='tab10', legend=False, s=10)
plt.title("DBSCAN Clustering – subject_type")
plt.tight_layout()
plt.savefig("src/outputs/dbscan_subject_type_labeled.png")
plt.close()

# Visualização HDBSCAN
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='hdbscan_cluster', data=df_result, palette='tab10', legend=False, s=10)
plt.title("HDBSCAN Clustering – subject_type")
plt.tight_layout()
plt.savefig("src/outputs/hdbscan_subject_type_labeled.png")
plt.close()
