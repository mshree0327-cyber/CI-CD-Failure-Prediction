"""Reproducible random and chronological 70:15:15 evaluation.

Random experiment: stratified 70/15/15 split on all build-level rows.
Chronological experiment: sort timestamp-matched rows by build_timestamp,
then allocate earliest 70%, next 15%, latest 15%.

A separate Random Forest is trained for each experiment. Validation data is
reported separately; final test metrics use only the held-out test partition.
The random-training model bundle is saved for the existing FastAPI predictor.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from features import PREBUILD_FEATURES, TARGET_NAME, BUILD_ID

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BACKEND_DIR = Path(__file__).resolve().parent
RANDOM_DATA = DATA_DIR / "travistorrent_build_level_prebuild.csv"
CHRONO_DATA = DATA_DIR / "timestamp_matched_build_level.csv"
MODEL_OUT = BACKEND_DIR / "random_forest_model.joblib"
RESULT_OUT = DATA_DIR / "evaluation_70_15_15.json"
SEED = 42

def prepare_frames(train_df, val_df, test_df):
    """Fit missing-value/category handling on train only; align other splits."""
    cat = "git_prev_commit_resolution_status"
    numeric = [c for c in PREBUILD_FEATURES if c != cat]
    train = train_df[PREBUILD_FEATURES].copy()
    val = val_df[PREBUILD_FEATURES].copy()
    test = test_df[PREBUILD_FEATURES].copy()

    medians = {}
    for col in numeric:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        val[col] = pd.to_numeric(val[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")
        med = train[col].median()
        medians[col] = float(med) if pd.notna(med) else 0.0
        train[col] = train[col].fillna(medians[col])
        val[col] = val[col].fillna(medians[col])
        test[col] = test[col].fillna(medians[col])

    for frame in (train, val, test):
        frame[cat] = frame[cat].astype("string").fillna("__MISSING__").astype(str)

    train_enc = pd.get_dummies(train, columns=[cat], dtype=int)
    val_enc = pd.get_dummies(val, columns=[cat], dtype=int).reindex(columns=train_enc.columns, fill_value=0)
    test_enc = pd.get_dummies(test, columns=[cat], dtype=int).reindex(columns=train_enc.columns, fill_value=0)
    return train_enc, val_enc, test_enc, medians, list(train_enc.columns), cat

def split_metrics(model, X, y, name):
    pred = model.predict(X)
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    failure_prob = probabilities[:, classes.index(0)]
    actual_failure = (y.astype(int) == 0).astype(int)
    predicted_failure = (pred == 0).astype(int)
    if y.nunique() < 2:
        auc = None
    else:
        auc = float(roc_auc_score(actual_failure, failure_prob))
    # Matrix order is explicitly Success, Failure (1, 0).
    cm = confusion_matrix(y.astype(int), pred.astype(int), labels=[1, 0]).tolist()
    return {
        "partition": name,
        "records": int(len(y)),
        "class_distribution": {
            "success": int((y.astype(int) == 1).sum()),
            "failure": int((y.astype(int) == 0).sum())
        },
        "accuracy": float(accuracy_score(y, pred)),
        "failure_precision": float(precision_score(actual_failure, predicted_failure, zero_division=0)),
        "failure_recall": float(recall_score(actual_failure, predicted_failure, zero_division=0)),
        "failure_f1": float(f1_score(actual_failure, predicted_failure, zero_division=0)),
        "failure_roc_auc": auc,
        "confusion_matrix_labels_success_then_failure": cm,
        "classification_report": classification_report(y, pred, labels=[0,1],
            target_names=["Failure","Success"], output_dict=True, zero_division=0)
    }

def make_model():
    return RandomForestClassifier(
        n_estimators=150, random_state=SEED, class_weight="balanced", n_jobs=-1
    )

def run_experiment(df, name, mode):
    df = df.copy()
    df[TARGET_NAME] = pd.to_numeric(df[TARGET_NAME], errors="coerce")
    df = df.dropna(subset=[TARGET_NAME])
    df = df[df[TARGET_NAME].isin([0, 1])].reset_index(drop=True)
    missing = [c for c in PREBUILD_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required 19 features: {missing}")

    if mode == "random":
        train_df, temp_df = train_test_split(
            df, test_size=0.30, random_state=SEED,
            stratify=df[TARGET_NAME].astype(int)
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=SEED,
            stratify=temp_df[TARGET_NAME].astype(int)
        )
    else:
        if "build_timestamp" not in df.columns:
            raise ValueError("Chronological data must contain build_timestamp.")
        df["build_timestamp"] = pd.to_datetime(df["build_timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["build_timestamp"]).sort_values(
            ["build_timestamp", BUILD_ID]
        ).reset_index(drop=True)
        n = len(df)
        i1, i2 = int(n * 0.70), int(n * 0.85)
        train_df, val_df, test_df = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError(f"{name}: one of the 70/15/15 partitions is empty.")
    for part_name, part in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        if part[TARGET_NAME].nunique() < 2:
            raise ValueError(
                f"{name} {part_name} partition contains only one class. "
                "AUC and reliable two-class evaluation cannot be computed. "
                "Do not silently shuffle chronological data to fix this."
            )

    Xtr, Xval, Xte, medians, encoded_features, cat = prepare_frames(train_df, val_df, test_df)
    ytr = train_df[TARGET_NAME].astype(int)
    yval = val_df[TARGET_NAME].astype(int)
    yte = test_df[TARGET_NAME].astype(int)
    model = make_model()
    model.fit(Xtr, ytr)
    return {
        "experiment": name,
        "split_method": mode,
        "random_state": SEED if mode == "random" else None,
        "feature_count_before_encoding": len(PREBUILD_FEATURES),
        "features": PREBUILD_FEATURES,
        "encoded_feature_count": len(encoded_features),
        "train": {"records": len(train_df), "class_distribution": {
            "success": int((ytr == 1).sum()), "failure": int((ytr == 0).sum())}},
        "validation_metrics": split_metrics(model, Xval, yval, "validation"),
        "test_metrics": split_metrics(model, Xte, yte, "held_out_test"),
        "_model": model, "_encoded_features": encoded_features,
        "_medians": medians, "_categorical_feature": cat
    }

def main():
    if not RANDOM_DATA.exists():
        raise FileNotFoundError(f"Missing {RANDOM_DATA}")
    if not CHRONO_DATA.exists():
        raise FileNotFoundError(f"Missing {CHRONO_DATA}")

    random_df = pd.read_csv(RANDOM_DATA, low_memory=False)
    chrono_df = pd.read_csv(CHRONO_DATA, low_memory=False)

    random_result = run_experiment(random_df, "Random stratified 70:15:15", "random")
    chrono_result = run_experiment(chrono_df, "Chronological 70:15:15", "chronological")

    # Keep the API's existing prediction contract: production model trained
    # only on the random experiment's 70% training partition.
    bundle = {
        "model": random_result["_model"],
        "features": random_result["_encoded_features"],
        "raw_features": PREBUILD_FEATURES,
        "categorical_feature": random_result["_categorical_feature"],
        "numeric_medians": random_result["_medians"],
        "categorical_levels": [c for c in random_result["_encoded_features"]
                               if c.startswith(random_result["_categorical_feature"] + "_")],
        "training_split": "random stratified 70:15:15; model fitted on training partition only",
        "random_state": SEED
    }
    joblib.dump(bundle, MODEL_OUT)

    def public_result(r):
        return {k:v for k,v in r.items() if not k.startswith("_")}

    output = {
        "title": "CI/CD failure prediction: 70:15:15 evaluation",
        "target": TARGET_NAME,
        "positive_class_for_failure_metrics": 0,
        "confusion_matrix_order": ["Success", "Failure"],
        "feature_policy": "Exactly 19 audited pre-build features; target, project identity, build ID and timestamp excluded from model inputs.",
        "random_experiment": public_result(random_result),
        "chronological_experiment": public_result(chrono_result),
        "notes": [
            "Validation and held-out test partitions are not used to fit the model.",
            "Random experiment uses stratified random splitting.",
            "Chronological experiment uses timestamp-matched records sorted by build_timestamp; earliest 70%, next 15%, latest 15%.",
            "Chronological results apply only to records in timestamp_matched_build_level.csv.",
            "The production API model bundle is trained on the random experiment training partition only."
        ]
    }
    RESULT_OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved model: {MODEL_OUT}")
    print(f"Saved evaluation: {RESULT_OUT}")
    for key in ["random_experiment", "chronological_experiment"]:
        result = output[key]
        print("\\n" + result["experiment"])
        print("Train:", result["train"]["records"],
              "Validation:", result["validation_metrics"]["records"],
              "Test:", result["test_metrics"]["records"])
        print("Test accuracy:", round(result["test_metrics"]["accuracy"] * 100, 2), "%")
        print("Test failure F1:", round(result["test_metrics"]["failure_f1"] * 100, 2), "%")
        print("Test confusion matrix:", result["test_metrics"]["confusion_matrix_labels_success_then_failure"])

if __name__ == "__main__":
    main()
