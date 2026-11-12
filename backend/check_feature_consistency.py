import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "travistorrent_100k_all_features.csv"

df = pd.read_csv(DATA_FILE)

print("========== FEATURE CONSISTENCY CHECK ==========")

# Exclude identifier and target
exclude = ["tr_build_id", "build_success"]
feature_columns = [
    col for col in df.columns
    if col not in exclude
]

# Number of unique values for each feature within each build
unique_counts = df.groupby("tr_build_id")[feature_columns].nunique()

# Count build-feature combinations where more than one value exists
inconsistent = (unique_counts > 1).sum()

print("\nFeatures with inconsistent values within builds:")

print(inconsistent[inconsistent > 0].sort_values(ascending=False))

print("\nTotal inconsistent feature types:")
print((inconsistent > 0).sum())

# Check first build
example_id = df["tr_build_id"].iloc[0]

print("\nExample build:", example_id)

print(
    df[df["tr_build_id"] == example_id][
        ["tr_build_id"] + feature_columns
    ].to_string(index=False)
)