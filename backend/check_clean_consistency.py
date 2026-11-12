import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "travistorrent_100k_clean.csv"

df = pd.read_csv(DATA_FILE)

print("========== CLEAN FEATURE CONSISTENCY ==========")

feature_columns = [
    col for col in df.columns
    if col not in ["tr_build_id", "build_success"]
]

unique_counts = df.groupby("tr_build_id")[feature_columns].nunique()

inconsistent = (unique_counts > 1).sum()

print("\nFeatures that vary within a build:")

print(inconsistent[inconsistent > 0].sort_values(ascending=False))

print("\nNumber of inconsistent clean features:")
print((inconsistent > 0).sum())

print("\nNumber of builds:")
print(df["tr_build_id"].nunique())