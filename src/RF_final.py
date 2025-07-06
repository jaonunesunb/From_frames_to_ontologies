import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

# Carregar dataset
df = pd.read_csv("src/outputs/paired_train_labeled_cleaned.csv")

# Target: escolha entre 'subject_type' ou 'predicate'
target_col = 'subject_type'
# target_col = 'predicate'

# Remover classes com menos de 25 instâncias
class_counts = df[target_col].value_counts()
valid_classes = class_counts[class_counts >= 25].index
df = df[df[target_col].isin(valid_classes)]

# Separar features e target
X = df.drop(columns=[target_col])
y = df[target_col]

# Label Encoding em colunas categóricas
categorical_cols = X.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Codificar target
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y)

# Normalizar colunas numéricas
numeric_cols = X.select_dtypes(include='number').columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Oversampling para balancear classes
ros = RandomOverSampler(random_state=42)
X_balanced, y_balanced = ros.fit_resample(X, y_encoded)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
)

# Treinamento
clf = RandomForestClassifier(n_estimators=100, max_depth=30, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Avaliação
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.4f}")

# Corrigir os labels para apenas os presentes no modelo
labels_names = y_encoder.inverse_transform(clf.classes_)
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix – Random Forest (max_depth=30)")
plt.tight_layout()
plt.savefig("src/outputs/confusion_matrix_rf.png")
plt.close()

