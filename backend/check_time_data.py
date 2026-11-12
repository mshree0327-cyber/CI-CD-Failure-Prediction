import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "travistorrent_100k_all_features.csv"

df = pd.read_csv(DATA_FILE)

print("========== TIME DATA INSPECTION ==========")

time_columns = [
    "build_year",
    "build_month",
    "build_hour",
    "build_day_of_week",
    "tr_build_id",
    "tr_build_number"
]

for col in time_columns:
    if col in df.columns:
        print(f"\n--- {col} ---")
        print(df[col].head(10).to_string(index=False))
        print("Unique values:", df[col].nunique())

print("\nFirst 10 build years:")
print(df["build_year"].head(10).tolist())

print("\nLast 10 build years:")
print(df["build_year"].tail(10).tolist())