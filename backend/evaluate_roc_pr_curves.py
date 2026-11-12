"""ROC and Precision-Recall curve analysis on the random 70:15:15 held-out test set.

The production Random Forest is loaded without retraining. Failure is treated
as the positive class: build_success == 0.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, auc, confusion_matrix
)
from features import PREBUILD_FEATURES, TARGET_NAME

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "travistorrent_build_level_prebuild.csv"
MODEL_FILE = Path(__file__).resolve().parent / "random_forest_model.joblib"
JSON_OUT = ROOT / "data" / "roc_pr_curve_metrics.json"
ROC_PNG = ROOT / "data" / "roc_curve.png"
PR_PNG = ROOT / "data" / "precision_recall_curve.png"
SEED = 42

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
    X_test = pd.get_dummies(X_test, columns=[cat], dtype=int).reindex(
        columns=bundle["features"], fill_value=0
    )

    # Failure is positive class for ROC and PR analysis.
    y_failure = (test_df[TARGET_NAME].astype(int).to_numpy() == 0).astype(int)
    model = bundle["model"]
    probabilities = model.predict_proba(X_test)
    failure_probability = probabilities[:, list(model.classes_).index(0)]

    fpr, tpr, roc_thresholds = roc_curve(y_failure, failure_probability)
    precision, recall, pr_thresholds = precision_recall_curve(y_failure, failure_probability)
    roc_auc_value = float(roc_auc_score(y_failure, failure_probability))
    average_precision = float(average_precision_score(y_failure, failure_probability))
    pr_auc_trapezoid = float(auc(recall, precision))
    prevalence = float(y_failure.mean())

    plt.figure(figsize=(7.5, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"Random Forest (ROC-AUC = {roc_auc_value:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Failure Recall)")
    plt.title("ROC Curve – Failure as Positive Class")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(ROC_PNG, dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 6))
    plt.plot(recall, precision, linewidth=2, label=f"Random Forest (AP = {average_precision:.4f})")
    plt.axhline(prevalence, linestyle="--", linewidth=1,
                label=f"Failure prevalence baseline = {prevalence:.4f}")
    plt.xlabel("Recall (Failure)")
    plt.ylabel("Precision (Failure)")
    plt.title("Precision–Recall Curve – Failure as Positive Class")
    plt.legend(loc="best")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PR_PNG, dpi=180, bbox_inches="tight")
    plt.close()

    result = {
        "title": "ROC and Precision-Recall curve analysis",
        "evaluation_partition": "Random stratified 70:15:15 held-out test",
        "split_random_state": SEED,
        "train_records": int(len(train_df)),
        "validation_records": int(len(val_df)),
        "test_records": int(len(test_df)),
        "positive_class": "Failure",
        "positive_class_encoding": "build_success == 0",
        "failure_records": int(y_failure.sum()),
        "success_records": int((1-y_failure).sum()),
        "failure_prevalence": prevalence,
        "roc_auc": roc_auc_value,
        "average_precision": average_precision,
        "precision_recall_auc_trapezoid": pr_auc_trapezoid,
        "roc_curve_image": "/api/evaluation-curves/roc",
        "precision_recall_curve_image": "/api/evaluation-curves/precision-recall",
        "notes": [
            "The existing Random Forest model was loaded; it was not retrained.",
            "Curves are computed only on the held-out random test partition.",
            "Failure (build_success=0) is treated as the positive class.",
            "ROC-AUC summarizes ranking discrimination across thresholds.",
            "Average Precision summarizes the precision-recall curve and is useful when class frequencies are imbalanced.",
            "The horizontal precision-recall reference line is the failure prevalence in this test partition."
        ]
    }
    JSON_OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("ROC/PR analysis completed.")
    print(f"Train / validation / test: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"Failure / success in test: {int(y_failure.sum())} / {int((1-y_failure).sum())}")
    print(f"ROC-AUC: {roc_auc_value:.6f}")
    print(f"Average Precision: {average_precision:.6f}")
    print(f"PR-AUC (trapezoidal): {pr_auc_trapezoid:.6f}")
    print("Saved:", JSON_OUT)
    print("Saved:", ROC_PNG)
    print("Saved:", PR_PNG)

if __name__ == "__main__":
    main()
