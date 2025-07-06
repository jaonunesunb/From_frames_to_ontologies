import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# 1) Carrega dados do CSV emparelhado
print('Carregando paired_train_labeled.csv...')
df = pd.read_csv('paired_train_labeled.csv')

# 2) Define a classe alvo para classificação binária
TARGET_CLASS = 'Interacts'  # altere para a classe que deseja prever
print(f"Alvo binário: 1 = {TARGET_CLASS}, 0 = outras classes")

# 3) Cria rótulo binário: 1 se predicate == TARGET_CLASS, senão 0
df['target'] = (df['predicate'] == TARGET_CLASS).astype(int)

# 4) Define colunas a remover (metadados e IDs)
drop_cols = ['video', 'event_id', 'predicate', 'actorA_id', 'actorB_id']

# 5) Separa features e alvo
X = df.drop(columns=drop_cols + ['target'])
y = df['target']

# 6) Seleciona apenas colunas numéricas
X_num = X.select_dtypes(include=[np.number])
features = X_num.columns.tolist()
print('Features numéricas usadas:', features)

# 7) Balanceamento com SMOTE (sintetiza exemplos da classe minoritária)
print('Aplicando SMOTE para balanceamento dos rótulos binários...')
over = SMOTE(random_state=42)
X_res, y_res = over.fit_resample(X_num, y)
print(f'Formato balanceado: X={X_res.shape}, y={y_res.shape}')

# 8) Monta DataFrame balanceado e salva para treino binário
balanced_df = pd.concat([
    pd.DataFrame(X_res, columns=features),
    y_res.rename('target')
], axis=1)
balanced_df.to_csv(f'paired_train_labeled_binary_{TARGET_CLASS}_balanced.csv', index=False)
print(f'Dados balanceados salvos em paired_train_labeled_binary_{TARGET_CLASS}_balanced.csv')
