import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# 1) Carrega dados
print('Carregando train_labeled.csv...')
df = pd.read_csv('train_labeled.csv')

# Colunas fixas a descartar: metadados e identificadores
base_drop = ['video', 'actor_id', 'frame']

# Alvos a treinar
targets = ['predicate', 'subject_type']

# Espaço de busca comum para RandomForest
dist_params = {
    'rf__n_estimators': [50, 100, 150, 200, 250, 300],
    'rf__max_depth': [None] + list(range(5, 55, 5)),
    'rf__min_samples_split': list(range(2, 21)),
}

# Loop de treinamento para cada target
for target in targets:
    print(f"\nTreinando modelo para target '{target}'...")

    # 2) Prepara X e y para o target atual
    drop_targets = [t for t in targets if t != target]
    # Remove colunas de metadados e as colunas de outros targets
    X = df.drop(columns=base_drop + drop_targets + [target])
    y = df[target]

    # 3) Define pipeline e RandomizedSearchCV
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=dist_params,
        n_iter=50,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    # 4) Executa busca de hiperparâmetros
    print(f"Iniciando busca de hiperparâmetros para '{target}'...")
    search.fit(X, y)

    # 5) Exibe resultados
    print(f"Melhor F1_macro (CV) para '{target}': {search.best_score_:.4f}")
    print(f"Melhores hiperparâmetros para '{target}': {search.best_params_}")

    # 6) Treina modelo final e salva
    best_model = search.best_estimator_
    print(f"Treinando modelo final para '{target}'...")
    best_model.fit(X, y)
    model_filename = f'rf_virat_{target}_optimized.joblib'
    joblib.dump(best_model, model_filename)
    print(f"Modelo salvo em {model_filename}")

# 7) (Opcional) Prever interações entre dois subject_types:
#    - Construir dataset de pares de atores por evento
#    - Gerar features relacionais (e.g., diferença de posição, duração conjunta)\#    - Definir y como concatenação de dois subject_type labels
#    - Treinar pipeline similar ao acima
