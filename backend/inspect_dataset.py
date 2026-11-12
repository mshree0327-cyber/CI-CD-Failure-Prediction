import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "travistorrent_100k_clean.csv"

df = pd.read_csv(DATA_FILE)

print("========== DATASET INSPECTION ==========")

print("\nShape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nTarget Distribution:")
print(df["build_success"].value_counts())

print("\nFirst 5 Rows:")
print(df.head())