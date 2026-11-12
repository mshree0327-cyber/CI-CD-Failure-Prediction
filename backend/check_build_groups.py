import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "travistorrent_100k_all_features.csv"

df = pd.read_csv(DATA_FILE)

print("========== BUILD GROUP CHECK ==========")

build_counts = df.groupby("tr_build_id").size()

print("\nNumber of unique builds:")
print(build_counts.size)

print("\nRows per build - statistics:")
print(build_counts.describe())

print("\nBuilds with more than 1 row:")
print((build_counts > 1).sum())

print("\nMaximum rows for one build:")
print(build_counts.max())

target_counts = df.groupby("tr_build_id")["build_success"].nunique()

print("\nBuilds having multiple target values:")
print((target_counts > 1).sum())

print("\nExample build:")
example_id = df["tr_build_id"].iloc[0]

print(
    df[df["tr_build_id"] == example_id][
        ["tr_build_id", "tr_build_number", "build_year",
         "build_month", "build_hour", "build_success"]
    ].to_string(index=False)
)