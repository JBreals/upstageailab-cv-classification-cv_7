import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def dfsplit(df, split_ratio = 0.2):
    train_df, val_df = train_test_split(df, test_size=split_ratio, random_state=42, stratify=df['target'])
    train_df['train'] = 1
    val_df['train'] = 0

    df = pd.concat([train_df, val_df], ignore_index=True)
    df.to_csv(os.path.join(ROOT_DIR, "data", "train_split.csv"), index=False)
    return df

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(ROOT_DIR, "data", "train.csv"))
    df = dfsplit(df, 0.2)
    print(df[df['train'] == 1]['target'].value_counts())
    print(df[df['train'] == 0]['target'].value_counts())

