"""Per-project performance analysis on the random held-out test partition.

This is descriptive subgroup analysis: the Random Forest is evaluated on
random-test builds and metrics are grouped by gh_project_name. It is NOT an
unseen-project evaluation; use evaluate_project_grouped.py for that experiment.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from features import PREBUILD_FEATURES, TARGET_NAME

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "travistorrent_build_level_prebuild.csv"
MODEL_FILE = Path(__file__).resolve().parent / "random_forest_model.joblib"
OUTPUT_FILE = ROOT / "data" / "project_level_performance.json"
SEED = 42
MIN_BUILDS_FLAG = 10

def main():
    if not DATA_FILE.exists() or not MODEL_FILE.exists():
        raise FileNotFoundError("Dataset or trained model bundle is missing.")
    bundle = joblib.load(MODEL_FILE)
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df[TARGET_NAME] = pd.to_numeric(df[TARGET_NAME], errors="coerce")
    df = df.dropna(subset=[TARGET_NAME, "gh_project_name"])
    df = df[df[TARGET_NAME].isin([0, 1])].reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df[TARGET_NAME].astype(int)
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED,
        stratify=temp_df[TARGET_NAME].astype(int)
    )

    cat = bundle["categorical_feature"]
    X = test_df[PREBUILD_FEATURES].copy()
    medians = bundle.get("numeric_medians", {})
    for col in PREBUILD_FEATURES:
        if col != cat:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(medians.get(col, 0.0))
    X[cat] = X[cat].astype("string").fillna("__MISSING__").astype(str)
    X = pd.get_dummies(X, columns=[cat], dtype=int).reindex(
        columns=bundle["features"], fill_value=0
    )

    y = test_df[TARGET_NAME].astype(int)
    pred = bundle["model"].predict(X)
    prob = bundle["model"].predict_proba(X)
    class_order = list(bundle["model"].classes_)
    failure_prob = prob[:, class_order.index(0)]

    scored = test_df[["gh_project_name", TARGET_NAME]].copy()
    scored["prediction"] = pred
    scored["failure_probability"] = failure_prob
    rows = []
    for project, part in scored.groupby("gh_project_name", dropna=False):
        actual = part[TARGET_NAME].astype(int).to_numpy()
        predicted = part["prediction"].astype(int).to_numpy()
        actual_failure = (actual == 0).astype(int)
        predicted_failure = (predicted == 0).astype(int)
        cm = confusion_matrix(actual, predicted, labels=[1, 0]).tolist()
        n = len(part)
        rows.append({
            "project": str(project),
            "builds": int(n),
            "successes": int((actual == 1).sum()),
            "failures": int((actual == 0).sum()),
            "accuracy": float(accuracy_score(actual, predicted)),
            "failure_precision": float(precision_score(actual_failure, predicted_failure, zero_division=0)),
            "failure_recall": float(recall_score(actual_failure, predicted_failure, zero_division=0)),
            "failure_f1": float(f1_score(actual_failure, predicted_failure, zero_division=0)),
            "confusion_matrix_success_then_failure": cm,
            "mean_failure_probability": float(part["failure_probability"].mean()),
            "small_sample_warning": bool(n < MIN_BUILDS_FLAG)
        })
    rows.sort(key=lambda r: (r["builds"], r["project"]), reverse=True)

    output = {
        "title": "Project-level performance analysis",
        "evaluation_method": "Per-project subgroup metrics on the random stratified held-out test partition",
        "source_partition": "Random stratified 70:15:15 held-out test",
        "train_records": int(len(train_df)),
        "validation_records": int(len(val_df)),
        "test_records": int(len(test_df)),
        "project_count_in_test": int(len(rows)),
        "minimum_builds_for_small_sample_warning": MIN_BUILDS_FLAG,
        "features_used": len(PREBUILD_FEATURES),
        "projects": rows,
        "notes": [
            "The existing production Random Forest is loaded; it is not retrained.",
            "Project identity is used only to group evaluation results, never as a model input.",
            "These are per-project metrics on a random held-out test set, not an unseen-project test.",
            "Metrics for projects with fewer than 10 test builds are flagged as small-sample and should be interpreted cautiously.",
            "Projects with no actual failures can have zero failure precision/recall/F1; these metrics do not imply the model is poor for that project without considering class support."
        ]
    }
    OUTPUT_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Saved:", OUTPUT_FILE)
    print("Random held-out test records:", len(test_df))
    print("Projects represented in test:", len(rows))
    print("Projects with fewer than", MIN_BUILDS_FLAG, "test builds:", sum(r["small_sample_warning"] for r in rows))
    print("\nTop projects by test-build count:")
    for r in rows[:10]:
        print(f'{r["project"]}: n={r["builds"]}, accuracy={r["accuracy"]:.3f}, '
              f'failure_recall={r["failure_recall"]:.3f}, failure_f1={r["failure_f1"]:.3f}')

if __name__ == "__main__":
    main()
