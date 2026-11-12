import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "travistorrent_100k_all_features.csv"

df = pd.read_csv(DATA_FILE)

print("========== DATA ORDER CHECK ==========")

print("\nFirst 20 rows - time information:")
print(
    df[
        [
            "build_year",
            "build_month",
            "build_day_of_week",
            "build_hour",
            "tr_build_id"
        ]
    ].head(20).to_string(index=False)
)

print("\nLast 20 rows - time information:")
print(
    df[
        [
            "build_year",
            "build_month",
            "build_day_of_week",
            "build_hour",
            "tr_build_id"
        ]
    ].tail(20).to_string(index=False)
)

print("\nYear distribution:")
print(df["build_year"].value_counts().sort_index())

print("\nFirst build ID:", df["tr_build_id"].iloc[0])
print("Last build ID:", df["tr_build_id"].iloc[-1])