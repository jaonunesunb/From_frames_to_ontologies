import pandas as pd

def clean_subject_type(input_path, output_path):
    df = pd.read_csv(input_path)

    df['subject_type'] = df['subject_type'].replace('0', 'Unknown')

    print(f"Total rows: {len(df)}")
    print(f"Number of 'Unknown' subject_type after cleaning: {sum(df['subject_type'] == 'Unknown')}")

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    clean_subject_type("src/outputs/paired_train_labeled.csv", "src/outputs/paired_train_labeled_cleaned.csv")
