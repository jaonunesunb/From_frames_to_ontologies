import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

# Carregar dataset
df = pd.read_csv("src/outputs/paired_train_labeled_cleaned.csv")

# Escolher coluna-alvo
target_col = 'subject_type'
# target_col = 'predicate'

# Separar features e target
X = df.drop(columns=[target_col])
y = df[target_col]

# Label encoding em colunas categóricas
categorical_cols = X.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Codificar o target
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y)

# Normalizar colunas numéricas
numeric_cols = X.select_dtypes(include='number').columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Selecionar subconjunto pequeno (sem stratify ainda!)
X_small, _, y_small, _ = train_test_split(X, y_encoded, train_size=2000, random_state=42)

# Aplicar oversampling no subconjunto
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_small, y_small)

# Agora sim: split com stratify
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# Treinar SVM
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_clf.fit(X_train, y_train)

# Avaliação
y_pred = svm_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia SVM: {accuracy:.4f}")

with open("src/outputs/accuracy_svm.txt", "w") as f:
    f.write(f"Acurácia SVM (RBF kernel): {accuracy:.4f}\n")

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred, labels=svm_clf.classes_)
labels_names = y_encoder.inverse_transform(svm_clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_names)
disp.plot(xticks_rotation=45, cmap='Greens')
plt.title('Confusion Matrix – SVM (RBF Kernel)')
plt.tight_layout()
plt.savefig("src/outputs/confusion_matrix_svm.png")
plt.close()
