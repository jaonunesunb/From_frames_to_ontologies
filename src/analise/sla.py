import pandas as pd

df = pd.read_csv("src/outputs/paired_train_labeled.csv")
# Mostrar valores únicos
print(df['subject_type'].unique())

# Contar quantos são '0'
zeros = df[df['subject_type'] == '0']
print(f"Total de registros com subject_type == '0': {len(zeros)}")

# Exibir amostras
print(zeros.head())
