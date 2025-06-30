import pandas as pd
from pathlib import Path
from feature_extraction import extract_features_from_video
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os

# Caminho da lista de vídeos rotulados
with open("videos_completos.txt", "r") as f:
    video_names = [line.strip() for line in f.readlines()]

# Define onde estão as anotações
annotations_dirs = [
    Path("viratannotations/train"),
    Path("viratannotations/validate")
]

# Função para encontrar o caminho correto do vídeo
def get_annotation_paths(video_name):
    for d in annotations_dirs:
        a_path = d / f"{video_name}.activities.yml"
        g_path = d / f"{video_name}.geom.yml"
        r_path = d / f"{video_name}.regions.yml"
        if a_path.exists() and g_path.exists() and r_path.exists():
            return a_path, g_path, r_path
    return None, None, None

# Coletar features de todos os vídeos rotulados
all_features = []

for video in tqdm(video_names, desc="Extraindo features"):
    a, g, r = get_annotation_paths(video)
    if a and g and r:
        try:
            feats = extract_features_from_video(a, g, r)
            for f in feats:
                f["video"] = video
                all_features.append(f)
        except Exception as e:
            print(f"Erro no vídeo {video}: {e}")

# Transformar em DataFrame
df = pd.DataFrame(all_features)

# Codificar o rótulo (tipo de evento)
le = LabelEncoder()
df["label"] = le.fit_transform(df["type"])

# Selecionar colunas numéricas
X = df[["duration", "num_actors", "avg_width", "avg_height", "regions_in_event"]]
y = df["label"]

# Treinar modelo com validação cruzada
clf = RandomForestClassifier(random_state=42)
scores = cross_val_score(clf, X, y, cv=5)

print("Acurácias na validação cruzada:", scores)
print("Acurácia média:", scores.mean())

# Mostrar os rótulos
print("\nLabel mapping (classe → número):")
for label, code in zip(le.classes_, le.transform(le.classes_)):
    print(f"{code} → {label}")

# Salvar DataFrame em CSV
output_path = "dados_rotulados.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Features salvas em: {output_path}")
