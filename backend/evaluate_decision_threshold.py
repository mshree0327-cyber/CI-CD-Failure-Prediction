"""Validation-selected decision threshold analysis for CI/CD failure prediction.

Threshold is selected on validation data only by maximum Failure F1.
The selected threshold is then evaluated once on the untouched random test set.
The existing Random Forest is loaded; it is not retrained.
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
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from features import PREBUILD_FEATURES, TARGET_NAME

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "travistorrent_build_level_prebuild.csv"
MODEL_FILE = Path(__file__).resolve().parent / "random_forest_model.joblib"
JSON_OUT = ROOT / "data" / "decision_threshold_analysis.json"
PNG_OUT = ROOT / "data" / "decision_threshold_metrics.png"
SEED = 42
CANDIDATE_THRESHOLDS = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]

def encode_frame(frame, bundle):
    cat = bundle["categorical_feature"]
    X = frame[PREBUILD_FEATURES].copy()
    medians = bundle.get("numeric_medians", {})
    for col in PREBUILD_FEATURES:
        if col != cat:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(medians.get(col, 0.0))
    X[cat] = X[cat].astype("string").fillna("__MISSING__").astype(str)
    return pd.get_dummies(X, columns=[cat], dtype=int).reindex(
        columns=bundle["features"], fill_value=0
    )

def metrics_at_threshold(y_failure, prob_failure, threshold):
    pred_failure = (prob_failure >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_failure, pred_failure)),
        "failure_precision": float(precision_score(y_failure, pred_failure, zero_division=0)),
        "failure_recall": float(recall_score(y_failure, pred_failure, zero_division=0)),
        "failure_f1": float(f1_score(y_failure, pred_failure, zero_division=0)),
        "false_positives": int(((y_failure == 0) & (pred_failure == 1)).sum()),
        "false_negatives": int(((y_failure == 1) & (pred_failure == 0)).sum()),
        "true_positives": int(((y_failure == 1) & (pred_failure == 1)).sum()),
        "true_negatives": int(((y_failure == 0) & (pred_failure == 0)).sum()),
        "confusion_matrix_success_then_failure": confusion_matrix(
            y_failure, pred_failure, labels=[0, 1]
        ).tolist()
    }

def main():
    bundle = joblib.load(MODEL_FILE)
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df[TARGET_NAME] = pd.to_numeric(df[TARGET_NAME], errors="coerce")
    df = df.dropna(subset=[TARGET_NAME])
    df = df[df[TARGET_NAME].isin([0, 1])].reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED,
        stratify=df[TARGET_NAME].astype(int)
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED,
        stratify=temp_df[TARGET_NAME].astype(int)
    )
    model = bundle["model"]
    failure_class_col = list(model.classes_).index(0)
    val_prob = model.predict_proba(encode_frame(val_df, bundle))[:, failure_class_col]
    test_prob = model.predict_proba(encode_frame(test_df, bundle))[:, failure_class_col]
    y_val = (val_df[TARGET_NAME].astype(int).to_numpy() == 0).astype(int)
    y_test = (test_df[TARGET_NAME].astype(int).to_numpy() == 0).astype(int)

    validation_rows = [
        metrics_at_threshold(y_val, val_prob, t) for t in CANDIDATE_THRESHOLDS
    ]
    # Predeclared selection rule: maximize validation Failure F1.
    # In a tie, choose the higher threshold to reduce unnecessary failure alerts.
    selected = sorted(
        validation_rows,
        key=lambda row: (row["failure_f1"], row["threshold"]),
        reverse=True
    )[0]
    chosen_threshold = selected["threshold"]
    test_at_selected = metrics_at_threshold(y_test, test_prob, chosen_threshold)
    test_at_default = metrics_at_threshold(y_test, test_prob, 0.50)

    # Plot validation curves only; test metrics remain an independent final check.
    thresholds = [r["threshold"] for r in validation_rows]
    plt.figure(figsize=(8.5, 5.5))
    for key, label in [
        ("failure_precision", "Failure precision"),
        ("failure_recall", "Failure recall"),
        ("failure_f1", "Failure F1-score"),
        ("accuracy", "Accuracy")
    ]:
        plt.plot(thresholds, [r[key] for r in validation_rows], marker="o", label=label)
    plt.axvline(chosen_threshold, linestyle="--", linewidth=1.5,
                label=f"Selected threshold = {chosen_threshold:.2f}")
    plt.xlabel("Failure probability threshold")
    plt.ylabel("Metric")
    plt.ylim(0, 1.02)
    plt.title("Decision Threshold Analysis (Validation Set)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_OUT, dpi=180, bbox_inches="tight")
    plt.close()

    result = {
        "title": "Decision threshold analysis",
        "split": "Random stratified 70:15:15",
        "random_state": SEED,
        "positive_class": "Failure",
        "positive_class_encoding": "build_success == 0",
        "threshold_selection": {
            "partition": "validation",
            "criterion": "Maximum Failure F1-score across candidate thresholds",
            "tie_break": "Higher threshold",
            "selected_threshold": chosen_threshold,
            "selected_validation_metrics": selected
        },
        "candidate_thresholds": CANDIDATE_THRESHOLDS,
        "validation_records": int(len(val_df)),
        "test_records": int(len(test_df)),
        "validation_results": validation_rows,
        "held_out_test_results": {
            "selected_threshold": test_at_selected,
            "default_threshold_0_50": test_at_default
        },
        "roc_auc_test": float(roc_auc_score(y_test, test_prob)),
        "plot": "decision_threshold_metrics.png",
        "notes": [
            "The existing Random Forest was loaded and not retrained.",
            "The threshold was selected using validation data only.",
            "The held-out test set was not used to select the threshold.",
            "Failure is the positive class; a build is predicted Failure when its failure probability is greater than or equal to the threshold.",
            "The threshold selection criterion is validation Failure F1, not a universal operational optimum.",
            "Operational deployment may prefer a different threshold depending on the relative costs of missed failures and false alarms."
        ]
    }
    JSON_OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Decision threshold analysis completed.")
    print(f"Train / validation / test: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print("Selected threshold (validation maximum Failure F1):", f"{chosen_threshold:.2f}")
    print("Validation Failure F1:", f'{selected["failure_f1"]:.4f}')
    print("Held-out test at selected threshold:")
    for k in ["accuracy", "failure_precision", "failure_recall", "failure_f1",
              "false_positives", "false_negatives"]:
        print(f"  {k}: {test_at_selected[k]}")
    print("Default 0.50 test Failure F1:", f'{test_at_default["failure_f1"]:.4f}')
    print("Saved:", JSON_OUT)
    print("Saved:", PNG_OUT)

if __name__ == "__main__":
    main()
