"""Error analysis on the random stratified 70:15:15 held-out test partition.

Target encoding: build_success=1 means Success; build_success=0 means Failure.
False positive = actual Success predicted Failure.
False negative = actual Failure predicted Success.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from features import PREBUILD_FEATURES, TARGET_NAME

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "travistorrent_build_level_prebuild.csv"
MODEL_FILE = Path(__file__).resolve().parent / "random_forest_model.joblib"
JSON_OUT = ROOT / "data" / "error_analysis.json"
CSV_OUT = ROOT / "data" / "misclassified_test_builds.csv"
SEED = 42
MAX_EXAMPLES = 100

def main():
    if not DATA_FILE.exists() or not MODEL_FILE.exists():
        raise FileNotFoundError("Dataset or trained Random Forest bundle is missing.")
    bundle = joblib.load(MODEL_FILE)
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df[TARGET_NAME] = pd.to_numeric(df[TARGET_NAME], errors="coerce")
    df = df.dropna(subset=[TARGET_NAME]).copy()
    df = df[df[TARGET_NAME].isin([0, 1])].reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED, stratify=df[TARGET_NAME].astype(int)
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED,
        stratify=temp_df[TARGET_NAME].astype(int)
    )

    cat = bundle["categorical_feature"]
    X_test = test_df[PREBUILD_FEATURES].copy()
    medians = bundle.get("numeric_medians", {})
    for col in PREBUILD_FEATURES:
        if col != cat:
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce").fillna(medians.get(col, 0.0))
    X_test[cat] = X_test[cat].astype("string").fillna("__MISSING__").astype(str)
    X_encoded = pd.get_dummies(X_test, columns=[cat], dtype=int).reindex(
        columns=bundle["features"], fill_value=0
    )

    y_true = test_df[TARGET_NAME].astype(int).to_numpy()
    model = bundle["model"]
    y_pred = model.predict(X_encoded).astype(int)
    probabilities = model.predict_proba(X_encoded)
    failure_col = list(model.classes_).index(0)
    failure_probability = probabilities[:, failure_col]

    # Explicitly map the positive class for error analysis as Failure.
    actual_failure = (y_true == 0)
    predicted_failure = (y_pred == 0)
    false_positive = (~actual_failure) & predicted_failure
    false_negative = actual_failure & (~predicted_failure)
    true_positive = actual_failure & predicted_failure
    true_negative = (~actual_failure) & (~predicted_failure)
    n = len(test_df)

    scored = pd.DataFrame({
        "actual_outcome": np.where(actual_failure, "Failure", "Success"),
        "predicted_outcome": np.where(predicted_failure, "Failure", "Success"),
        "error_type": np.select(
            [false_positive, false_negative, true_positive, true_negative],
            ["False Positive", "False Negative", "Correct Failure", "Correct Success"],
            default="Unknown"
        ),
        "failure_probability": failure_probability
    }, index=test_df.index)

    # Keep useful audit identifiers if present; never feed these to the model.
    for col in ["tr_build_id", "gh_project_name", "tr_job_id", "build_timestamp"]:
        if col in test_df.columns:
            scored[col] = test_df[col].astype(str).values
    for col in PREBUILD_FEATURES:
        scored[col] = test_df[col].values

    errors = scored[scored["error_type"].isin(["False Positive", "False Negative"])].copy()
    errors["failure_probability"] = errors["failure_probability"].round(6)
    errors.to_csv(CSV_OUT, index=False)

    def count(mask):
        return int(np.asarray(mask).sum())
    def pct(mask):
        return round(count(mask) / n * 100.0, 2) if n else 0.0

    # Compare numeric feature averages by error group descriptively.
    feature_comparison = []
    raw_test = test_df[PREBUILD_FEATURES].copy()
    for feature in PREBUILD_FEATURES:
        numeric = pd.to_numeric(raw_test[feature], errors="coerce")
        if numeric.notna().any():
            def avg(mask):
                vals = numeric.loc[np.asarray(mask, dtype=bool)]
                return float(vals.mean()) if len(vals) and vals.notna().any() else None
            feature_comparison.append({
                "feature": feature,
                "mean_false_positive": avg(false_positive),
                "mean_false_negative": avg(false_negative),
                "mean_correct_predictions": avg(true_positive | true_negative)
            })

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0]).tolist()
    examples = []
    for _, row in errors.head(MAX_EXAMPLES).iterrows():
        item = {
            "actual": row["actual_outcome"],
            "predicted": row["predicted_outcome"],
            "error_type": row["error_type"],
            "failure_probability": float(row["failure_probability"])
        }
        for col in ["tr_build_id", "gh_project_name"]:
            if col in row.index:
                item[col] = str(row[col])
        examples.append(item)

    result = {
        "title": "Held-out test error analysis",
        "evaluation_partition": "Random stratified 70:15:15 held-out test",
        "split_random_state": SEED,
        "train_records": int(len(train_df)),
        "validation_records": int(len(val_df)),
        "test_records": int(n),
        "target_definition": {"build_success": 1, "build_failure": 0},
        "error_definitions": {
            "false_positive": "Actual Success predicted as Failure",
            "false_negative": "Actual Failure predicted as Success",
            "true_positive": "Actual Failure correctly predicted as Failure",
            "true_negative": "Actual Success correctly predicted as Success"
        },
        "counts": {
            "false_positive": count(false_positive),
            "false_negative": count(false_negative),
            "correct_failure": count(true_positive),
            "correct_success": count(true_negative),
            "total_misclassified": count(false_positive) + count(false_negative)
        },
        "percentages_of_test_set": {
            "false_positive": pct(false_positive),
            "false_negative": pct(false_negative),
            "total_misclassified": round((count(false_positive) + count(false_negative)) / n * 100.0, 2)
        },
        "confusion_matrix_actual_success_then_failure": cm,
        "feature_mean_comparison": feature_comparison,
        "misclassified_examples": examples,
        "misclassified_csv": "misclassified_test_builds.csv",
        "notes": [
            "The existing production Random Forest was loaded; it was not retrained.",
            "Predictions and errors were calculated only on the held-out test partition.",
            "False Positive means a successful build was incorrectly flagged as Failure.",
            "False Negative means a failed build was incorrectly predicted as Success.",
            "Feature averages by error group are descriptive associations, not causal explanations.",
            "Build IDs and project names are included only for audit/reporting and are not model inputs."
        ]
    }
    JSON_OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Error analysis completed.")
    print("Train / validation / held-out test:", len(train_df), "/", len(val_df), "/", n)
    print("False positives (Success -> Failure):", count(false_positive), f"({pct(false_positive)}%)")
    print("False negatives (Failure -> Success):", count(false_negative), f"({pct(false_negative)}%)")
    print("Correct failures:", count(true_positive), "| Correct successes:", count(true_negative))
    print("Total misclassified:", result["counts"]["total_misclassified"])
    print("Saved:", JSON_OUT)
    print("Saved:", CSV_OUT)

if __name__ == "__main__":
    main()
