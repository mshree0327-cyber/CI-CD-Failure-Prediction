"""Bootstrap 95% confidence intervals for held-out test metrics.

Recreates the random stratified 70:15:15 split (seed 42), loads the existing
Random Forest bundle without retraining, and bootstraps the held-out test
predictions. Failure (build_success=0) is the positive class.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from features import PREBUILD_FEATURES, TARGET_NAME

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "travistorrent_build_level_prebuild.csv"
MODEL_FILE = Path(__file__).resolve().parent / "random_forest_model.joblib"
OUTPUT_FILE = ROOT / "data" / "bootstrap_confidence_intervals.json"
SEED = 42
N_BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 2026

def metric_values(y_true_failure, y_pred_failure, failure_probability):
    return {
        "accuracy": float(accuracy_score(y_true_failure, y_pred_failure)),
        "failure_precision": float(precision_score(y_true_failure, y_pred_failure, zero_division=0)),
        "failure_recall": float(recall_score(y_true_failure, y_pred_failure, zero_division=0)),
        "failure_f1": float(f1_score(y_true_failure, y_pred_failure, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_failure, failure_probability))
    }

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model bundle not found: {MODEL_FILE}")

    bundle = joblib.load(MODEL_FILE)
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df[TARGET_NAME] = pd.to_numeric(df[TARGET_NAME], errors="coerce")
    df = df.dropna(subset=[TARGET_NAME])
    df = df[df[TARGET_NAME].isin([0, 1])].reset_index(drop=True)

    missing = [c for c in PREBUILD_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing audited pre-build features: {missing}")

    # This is the same two-stage split used by evaluate_70_15_15.py.
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED,
        stratify=df[TARGET_NAME].astype(int)
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED,
        stratify=temp_df[TARGET_NAME].astype(int)
    )

    cat_feature = bundle["categorical_feature"]
    medians = bundle.get("numeric_medians", {})
    X_test = test_df[PREBUILD_FEATURES].copy()
    for col in PREBUILD_FEATURES:
        if col != cat_feature:
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
            X_test[col] = X_test[col].fillna(medians.get(col, 0.0))
    X_test[cat_feature] = X_test[cat_feature].astype("string").fillna("__MISSING__").astype(str)
    X_test = pd.get_dummies(X_test, columns=[cat_feature], dtype=int)
    X_test = X_test.reindex(columns=bundle["features"], fill_value=0)

    model = bundle["model"]
    probabilities = model.predict_proba(X_test)
    classes = list(model.classes_)
    failure_probability = probabilities[:, classes.index(0)]
    predicted_label = model.predict(X_test)
    y_true_failure = (test_df[TARGET_NAME].astype(int).to_numpy() == 0).astype(int)
    y_pred_failure = (predicted_label == 0).astype(int)

    point = metric_values(y_true_failure, y_pred_failure, failure_probability)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(y_true_failure)
    samples = {name: [] for name in point}
    accepted = 0
    for _ in range(N_BOOTSTRAPS):
        indices = rng.integers(0, n, size=n)
        yb = y_true_failure[indices]
        # ROC-AUC is undefined if a resample contains only one class.
        if np.unique(yb).size < 2:
            continue
        values = metric_values(
            yb, y_pred_failure[indices], failure_probability[indices]
        )
        for name, value in values.items():
            samples[name].append(value)
        accepted += 1

    intervals = {}
    for name, values in samples.items():
        low, high = np.percentile(np.asarray(values), [2.5, 97.5])
        intervals[name] = {
            "estimate": point[name],
            "ci_95_lower": float(low),
            "ci_95_upper": float(high),
            "bootstrap_std": float(np.std(values, ddof=1))
        }

    result = {
        "title": "Bootstrap 95% confidence intervals for held-out test metrics",
        "evaluation_partition": "Random stratified 70:15:15 held-out test",
        "split_random_state": SEED,
        "bootstrap_method": "Non-parametric percentile bootstrap over paired test records",
        "bootstrap_replicates_requested": N_BOOTSTRAPS,
        "bootstrap_replicates_used": accepted,
        "bootstrap_random_state": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "train_records": int(len(train_df)),
        "validation_records": int(len(val_df)),
        "test_records": int(n),
        "positive_class": "Failure",
        "positive_class_encoding": "build_success == 0",
        "test_class_distribution": {
            "success": int((y_true_failure == 0).sum()),
            "failure": int((y_true_failure == 1).sum())
        },
        "metrics": intervals,
        "interpretation": [
            "The point estimate is calculated on the complete held-out test partition.",
            "The 95% interval is the 2.5th to 97.5th percentile of bootstrap metric estimates.",
            "Each bootstrap resample draws test records with replacement and keeps actual labels, predictions and probabilities paired.",
            "The existing model is loaded and is not retrained.",
            "These intervals quantify test-sample uncertainty; they do not include uncertainty from retraining, hyperparameter selection or dataset shift."
        ]
    }
    OUTPUT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Bootstrap confidence interval analysis completed.")
    print(f"Train / validation / test: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"Bootstrap replicates used: {accepted} / {N_BOOTSTRAPS}")
    for name, item in intervals.items():
        print(f"{name}: {item['estimate']:.4f} (95% CI {item['ci_95_lower']:.4f}, {item['ci_95_upper']:.4f})")
    print("Saved:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
