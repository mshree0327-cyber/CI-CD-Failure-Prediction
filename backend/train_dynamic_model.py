"""Train/evaluate a leakage-aware dynamic-feature Random Forest.

Default input: data/timestamp_matched_build_level.csv
Default output: backend/random_forest_dynamic_model.joblib
Uses chronological holdout (earliest 80% train, latest 20% test).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score,
)

from dynamic_feature_config import (
    DYNAMIC_FEATURES, TARGET_NAME, TIMESTAMP_NAME,
    PROJECT_NAME, BUILD_ID_NAME, FEATURE_DEFINITIONS, EXCLUDED_FEATURES,
)


def evaluate(y_true, pred, prob):
    result = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "failure_precision": float(precision_score(y_true, pred, pos_label=0, zero_division=0)),
        "failure_recall": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        "failure_f1": float(f1_score(y_true, pred, pos_label=0, zero_division=0)),
        "confusion_matrix_labels_0_failure_1_success": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_true, pred, labels=[0, 1], target_names=["failure", "success"], output_dict=True, zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        # Positive class for ROC/PR is failure (0), so invert success probability.
        failure_probability = 1.0 - prob[:, 1]
        result["failure_roc_auc"] = float(roc_auc_score((y_true == 0).astype(int), failure_probability))
        result["failure_pr_auc"] = float(average_precision_score((y_true == 0).astype(int), failure_probability))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/timestamp_matched_build_level.csv")
    parser.add_argument("--model-out", default="backend/random_forest_dynamic_model.joblib")
    parser.add_argument("--metrics-out", default="backend/dynamic_model_metrics.json")
    parser.add_argument("--test-fraction", type=float, default=0.20)
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Copy timestamp_matched_build_level.csv into data/ or pass --input.")
    df = pd.read_csv(data_path)

    required = DYNAMIC_FEATURES + [TARGET_NAME, TIMESTAMP_NAME]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[TIMESTAMP_NAME] = pd.to_datetime(df[TIMESTAMP_NAME], utc=True, errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_NAME, TARGET_NAME]).copy()
    df[TARGET_NAME] = pd.to_numeric(df[TARGET_NAME], errors="coerce")
    df = df[df[TARGET_NAME].isin([0, 1])].copy()
    for col in DYNAMIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values([TIMESTAMP_NAME, BUILD_ID_NAME] if BUILD_ID_NAME in df.columns else [TIMESTAMP_NAME]).reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(f"Not enough timestamped rows for a holdout: {len(df)}")
    split_at = int(len(df) * (1.0 - args.test_fraction))
    if split_at <= 0 or split_at >= len(df):
        raise ValueError("Invalid test fraction; must leave non-empty train and test sets.")

    train_df, test_df = df.iloc[:split_at].copy(), df.iloc[split_at:].copy()
    X_train = train_df[DYNAMIC_FEATURES]
    X_test = test_df[DYNAMIC_FEATURES]
    y_train = train_df[TARGET_NAME].astype(int)
    y_test = test_df[TARGET_NAME].astype(int)

    # Median imputation values are learned only from the training period.
    medians = X_train.median(numeric_only=True).fillna(0.0)
    X_train = X_train.fillna(medians).fillna(0.0)
    X_test = X_test.fillna(medians).fillna(0.0)

    model = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced",
        n_jobs=-1, min_samples_leaf=2,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_metrics = {
        "accuracy": float(accuracy_score(y_test, dummy_pred)),
        "failure_precision": float(precision_score(y_test, dummy_pred, pos_label=0, zero_division=0)),
        "failure_recall": float(recall_score(y_test, dummy_pred, pos_label=0, zero_division=0)),
        "failure_f1": float(f1_score(y_test, dummy_pred, pos_label=0, zero_division=0)),
    }

    metrics = {
        "dataset": str(data_path),
        "rows_used": int(len(df)),
        "feature_count": len(DYNAMIC_FEATURES),
        "features": DYNAMIC_FEATURES,
        "excluded_features": EXCLUDED_FEATURES,
        "split_method": "Chronological holdout: earliest 80% train, latest 20% test",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_start": str(train_df[TIMESTAMP_NAME].min()),
        "train_end": str(train_df[TIMESTAMP_NAME].max()),
        "test_start": str(test_df[TIMESTAMP_NAME].min()),
        "test_end": str(test_df[TIMESTAMP_NAME].max()),
        "train_projects": int(train_df[PROJECT_NAME].nunique()) if PROJECT_NAME in train_df else None,
        "test_projects": int(test_df[PROJECT_NAME].nunique()) if PROJECT_NAME in test_df else None,
        "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "majority_class_baseline": dummy_metrics,
        "random_forest": evaluate(y_test, pred, prob),
        "feature_importance": {k: float(v) for k, v in sorted(zip(DYNAMIC_FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True)},
        "limitations": [
            "The supplied dictionary does not include the original feature-generation code.",
            "Several live feature calculations are operational approximations and are not proven equivalent to the historical dataset definitions.",
            "gh_sloc and test-density metrics are currently Python-only in the live extractor.",
            "code_churn_rate currently differs from the dictionary's lines/day unit and must be reconciled before claiming feature parity.",
            "gh_num_commits_on_files_touched may double-count a commit that touches multiple changed files.",
            "A chronological split is used; projects may occur in both periods, so this is not an unseen-project generalization test.",
            "The evaluation dataset contains only timestamp-matched records, not all 18,095 records.",
        ],
    }

    model_out = Path(args.model_out)
    metrics_out = Path(args.metrics_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "features": DYNAMIC_FEATURES,
        "raw_features": DYNAMIC_FEATURES,
        "categorical_feature": None,
        "numeric_medians": {k: float(v) for k, v in medians.items()},
        "target": TARGET_NAME,
        "feature_definitions": FEATURE_DEFINITIONS,
        "excluded_features": EXCLUDED_FEATURES,
        "evaluation": "chronological_holdout",
    }
    joblib.dump(bundle, model_out)
    metrics_out.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "model_saved": str(model_out),
        "metrics_saved": str(metrics_out),
        "rows_used": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "feature_count": len(DYNAMIC_FEATURES),
        "majority_baseline": dummy_metrics,
        "random_forest": {k: v for k, v in metrics["random_forest"].items() if k != "classification_report"},
        "warning": "Do not deploy until live feature definitions are reconciled with training feature calculations.",
    }, indent=2))

if __name__ == "__main__":
    main()
