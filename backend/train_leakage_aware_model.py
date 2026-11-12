"""Train the production pre-build model with a reproducible stratified split.

This script intentionally uses a random build-level split only. It does not
claim to be a chronological evaluation. Use a timestamped export for that.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from features import PREBUILD_FEATURES, TARGET_NAME, BUILD_ID

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INPUT = DATA / "travistorrent_build_level_prebuild.csv"
MODEL_OUT = Path(__file__).resolve().parent / "random_forest_model.joblib"
RESULT_OUT = DATA / "production_model_random_split_metrics.json"

def main():
    df = pd.read_csv(INPUT)
    X = df[PREBUILD_FEATURES].copy()
    y = df[TARGET_NAME].astype(int)
    # Fit category encoding from training data only.
    cat = "git_prev_commit_resolution_status"
    X[cat] = X[cat].astype("string").fillna("__MISSING__")
    X = pd.get_dummies(X, columns=[cat], dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=150, random_state=42, class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, list(model.classes_).index(0)]
    metrics = {
        "split": "stratified random build-level 80:20",
        "random_state": 42,
        "features_before_encoding": PREBUILD_FEATURES,
        "encoded_feature_names": list(X.columns),
        "accuracy": accuracy_score(y_test, pred),
        "failure_precision": precision_score(y_test, pred, pos_label=0, zero_division=0),
        "failure_recall": recall_score(y_test, pred, pos_label=0, zero_division=0),
        "failure_f1": f1_score(y_test, pred, pos_label=0, zero_division=0),
        "failure_roc_auc": roc_auc_score((y_test == 0).astype(int), probs),
        "confusion_matrix_labels_0_failure_1_success": confusion_matrix(y_test, pred, labels=[0,1]).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }
    # Store preprocessing metadata with the estimator; the API must use this.
    joblib.dump({"model": model, "features": list(X.columns),
                 "raw_features": PREBUILD_FEATURES, "categorical_feature": cat,
                 "categorical_levels": sorted(X.filter(like=cat+"_").columns.tolist())},
                MODEL_OUT)
    DATA.mkdir(parents=True, exist_ok=True)
    RESULT_OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in metrics.items() if k != "classification_report"}, indent=2))
    print(f"Model saved: {MODEL_OUT}")
    print(f"Metrics saved: {RESULT_OUT}")

if __name__ == "__main__":
    main()
