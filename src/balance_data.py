import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import argparse
import numpy as np

def load_data(csv_path="dados_rotulados.csv"):
    df = pd.read_csv(csv_path)
    return df

def undersample(df):
    min_count = df['type'].value_counts().min()
    dfs = [resample(g, replace=False, n_samples=min_count, random_state=42)
           for _, g in df.groupby('type')]
    return pd.concat(dfs)

def oversample_smote(df):
    from sklearn.preprocessing import LabelEncoder

    # Filtra apenas classes com >= 2 amostras (evita erro do SMOTE)
    valid_classes = df["type"].value_counts()
    valid_classes = valid_classes[valid_classes >= 2].index.tolist()
    df = df[df["type"].isin(valid_classes)]

    X = df[["duration", "num_actors", "avg_width", "avg_height", "regions_in_event"]]
    y = df["type"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    sm = SMOTE(random_state=42, k_neighbors=1)  # usa 1 vizinho só para evitar erro
    X_res, y_res = sm.fit_resample(X, y_encoded)

    df_res = pd.DataFrame(X_res, columns=X.columns)
    df_res["type"] = le.inverse_transform(y_res)
    return df_res

def binarize(df):
    comuns = df['type'].value_counts().nlargest(5).index.tolist()
    df["type_bin"] = df["type"].apply(lambda x: "normal" if x in comuns else "anormal")
    return df.drop(columns=["type"]).rename(columns={"type_bin": "type"})

def balance_data(strategy):
    df = load_data()

    if strategy == "undersample":
        df_bal = undersample(df)
    elif strategy == "oversample":
        df_bal = oversample_smote(df)
    elif strategy == "binarize":
        df_bal = binarize(df)
    else:
        raise ValueError("Estratégia inválida. Use 'undersample', 'oversample' ou 'binarize'.")

    df_bal.to_csv("dados_balanceados.csv", index=False)
    print(f"✅ Dados balanceados salvos em 'dados_balanceados.csv'\n")
    print(df_bal['type'].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", choices=["undersample", "oversample", "binarize"],
                        help="Estratégia de balanceamento a aplicar")
    args = parser.parse_args()

    balance_data(args.strategy)
