import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar dataset
df = pd.read_csv("src/outputs/paired_train_labeled_cleaned.csv")

# Mostrar info básica
print("Total rows:", len(df))

# Contagem das classes
subject_counts = df['subject_type'].value_counts()
predicate_counts = df['predicate'].value_counts()

# Gráfico de barras - Subject Types
plt.figure(figsize=(10, 5))
sns.barplot(x=subject_counts.index, y=subject_counts.values)
plt.xticks(rotation=45, ha='right')
plt.title('Distribution of Subject Types (Classes)')
plt.ylabel('Frequency')
plt.xlabel('Subject Type')
plt.tight_layout()
plt.show()

# Gráfico de barras - Predicates
plt.figure(figsize=(10, 5))
sns.barplot(x=predicate_counts.index, y=predicate_counts.values)
plt.xticks(rotation=45, ha='right')
plt.title('Distribution of Predicates (Relationships)')
plt.ylabel('Frequency')
plt.xlabel('Predicate')
plt.tight_layout()
plt.show()

# Heatmap de coocorrência entre subject_type e predicate
cooc_matrix = pd.crosstab(df['subject_type'], df['predicate'])

plt.figure(figsize=(12, 8))
sns.heatmap(cooc_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Co-occurrence Matrix: Subject Type vs Predicate")
plt.ylabel("Subject Type")
plt.xlabel("Predicate")
plt.tight_layout()
plt.show()
