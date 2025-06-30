import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("dados_balanceados.csv")

le = LabelEncoder()
df["label"] = le.fit_transform(df["type"])

X = df[["duration", "num_actors", "avg_width", "avg_height", "regions_in_event"]]
y_true = df["label"]

clf = RandomForestClassifier(random_state=42)
clf.fit(X, y_true)
y_pred = clf.predict(X)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(16, 12))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Matriz de Confusão - Modelo com Oversampling")
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.tight_layout()
plt.savefig("matriz_confusao_oversample.png", dpi=300)
plt.show()
