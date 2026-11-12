import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "travistorrent_build_level.csv"

TRAIN_FILE = BASE_DIR / "data" / "train_temporal.csv"
TEST_FILE = BASE_DIR / "data" / "test_temporal.csv"

df = pd.read_csv(DATA_FILE)

print("========== TEMPORAL SPLIT ==========")

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print("\nTotal builds:", len(df))
print("Training builds:", len(train_df))
print("Testing builds:", len(test_df))

print("\nTraining period:")
print("First build ID:", train_df["tr_build_id"].iloc[0])
print("Last build ID:", train_df["tr_build_id"].iloc[-1])

print("\nTesting period:")
print("First build ID:", test_df["tr_build_id"].iloc[0])
print("Last build ID:", test_df["tr_build_id"].iloc[-1])

print("\nTraining target:")
print(train_df["build_success"].value_counts())

print("\nTesting target:")
print(test_df["build_success"].value_counts())

print("\nFiles saved:")
print(TRAIN_FILE)
print(TEST_FILE)