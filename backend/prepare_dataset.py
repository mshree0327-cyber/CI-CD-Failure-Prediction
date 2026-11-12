"""Create one row per build using only audited pre-build features."""
from pathlib import Path
import pandas as pd
from features import PREBUILD_FEATURES, TARGET_NAME, BUILD_ID, METADATA_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SOURCE = DATA / "travistorrent_100k_all_features.csv"
OUTPUT = DATA / "travistorrent_build_level_prebuild.csv"

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Place source CSV at: {SOURCE}")
    df = pd.read_csv(SOURCE, low_memory=False)
    required = [BUILD_ID, TARGET_NAME] + PREBUILD_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Target and project identity must be consistent for each build.
    for col in [TARGET_NAME, "gh_project_name", "tr_build_number"]:
        if col in df.columns:
            bad = df.groupby(BUILD_ID, dropna=False)[col].nunique(dropna=False)
            bad = bad[bad > 1]
            if len(bad):
                raise ValueError(f"{col} is inconsistent in {len(bad)} build groups")

    # Features must be stable across jobs belonging to the same build.
    for col in PREBUILD_FEATURES:
        bad = df.groupby(BUILD_ID, dropna=False)[col].nunique(dropna=True)
        if (bad > 1).any():
            raise ValueError(f"Feature {col} varies within a build; aggregation needs review")

    keep = [BUILD_ID] + (["gh_project_name"] if "gh_project_name" in df.columns else []) + PREBUILD_FEATURES + [TARGET_NAME]
    build_df = df[keep].drop_duplicates(subset=[BUILD_ID], keep="first").copy()
    build_df = build_df.dropna(subset=[TARGET_NAME])
    build_df[TARGET_NAME] = pd.to_numeric(build_df[TARGET_NAME], errors="raise").astype(int)
    if not set(build_df[TARGET_NAME].unique()).issubset({0, 1}):
        raise ValueError("build_success must be binary 0/1")
    DATA.mkdir(parents=True, exist_ok=True)
    build_df.to_csv(OUTPUT, index=False)
    print(f"Job rows: {len(df):,}")
    print(f"Unique builds: {build_df[BUILD_ID].nunique():,}")
    print(f"Build-level rows saved: {len(build_df):,}")
    print(f"Pre-build features: {len(PREBUILD_FEATURES)}")
    print(f"Target distribution:\\n{build_df[TARGET_NAME].value_counts().to_string()}")
    print(f"Saved: {OUTPUT}")

if __name__ == "__main__":
    main()
