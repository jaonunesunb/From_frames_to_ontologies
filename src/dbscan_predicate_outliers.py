# src/dbscan_predicate_outliers.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN


# 1. Carrega o dataset (rotulado ou não)
df = pd.read_csv('new_train_labeled.csv')

# 2. Pré-processamento
for col in ['video', 'id', 'filename']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
df.dropna(inplace=True)

# 3. Separa features (se houver coluna 'predicate', ela é removida)
X = df.drop(columns=['predicate'], errors='ignore')

# 4. Identifica colunas numéricas e categóricas
numeric_cols     = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 5. Pré-processador: MinMaxScaler para numéricos e OneHot para categóricos
preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), numeric_cols),
    ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
])

# 6. Pipeline: pré-processamento + DBSCAN
pipeline = Pipeline([
    ('preproc', preprocessor),
    ('dbscan', DBSCAN(eps=0.5, min_samples=5, metric='euclidean'))
])

# 7. Ajusta o modelo de cluster (sem usar rótulos)
pipeline.fit(X)
labels = pipeline.named_steps['dbscan'].labels_

# 8. Detecta índices de outliers (label == -1)
outlier_indices = np.where(labels == -1)[0]

# 9. Salva lista de índices de outliers em arquivo
with open('dbscan_outliers_indices.txt', 'w') as f:
    for idx in outlier_indices:
        f.write(f"{idx}\n")

print(f"Detectados {len(outlier_indices)} outliers. Índices salvos em 'dbscan_outliers_indices.txt'.")

# 10. (Opcional) Salva DataFrame com cluster e flag de outlier
df_result = X.copy()
df_result['cluster'] = labels
df_result['is_outlier'] = (labels == -1)
df_result.to_csv('dbscan_predicate_clusters_with_outliers.csv', index=True)

