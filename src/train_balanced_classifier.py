import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Carregar dados balanceados
df = pd.read_csv("dados_balanceados.csv")

# Codificar rótulo
le = LabelEncoder()
df["label"] = le.fit_transform(df["type"])

# Features
X = df[["duration", "num_actors", "avg_width", "avg_height", "regions_in_event"]]
y = df["label"]

# Classificador
clf = RandomForestClassifier(random_state=42)
scores = cross_val_score(clf, X, y, cv=5)

# Resultados
print("Acurácias na validação cruzada:", scores)
print("Acurácia média:", scores.mean())

print("\nLabel mapping (classe → número):")
for label, code in zip(le.classes_, le.transform(le.classes_)):
    print(f"{code} → {label}")
