import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "travistorrent_100k_clean.csv"
OUTPUT_FILE = BASE_DIR / "data" / "travistorrent_build_level.csv"

df = pd.read_csv(DATA_FILE)

print("========== BUILD-LEVEL DATASET ==========")

# Keep the first occurrence of each build.
# Clean features are already verified to be identical
# within each build.
build_df = df.drop_duplicates(subset=["tr_build_id"], keep="first").copy()

build_df.to_csv(OUTPUT_FILE, index=False)

print("Original rows:", len(df))
print("Unique builds:", build_df["tr_build_id"].nunique())
print("Final rows:", len(build_df))
print("Final columns:", len(build_df.columns))

print("\nTarget distribution:")
print(build_df["build_success"].value_counts())

print("\nDataset saved to:")
print(OUTPUT_FILE)