import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "travistorrent_build_level.csv"

df = pd.read_csv(DATA_FILE)

print("========== BUILD ORDER VERIFICATION ==========")

print("\nFirst 10 build IDs:")
print(df["tr_build_id"].head(10).tolist())

print("\nLast 10 build IDs:")
print(df["tr_build_id"].tail(10).tolist())

print("\nBuild IDs increasing:")
print(df["tr_build_id"].is_monotonic_increasing)

print("\nUnique build IDs:")
print(df["tr_build_id"].nunique())

print("\nTotal rows:")
print(len(df))