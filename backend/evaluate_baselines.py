"""Baseline comparisons on the same held-out test partitions.

Majority baseline: always predicts the majority class in the training partition.
Previous-build baseline: for timestamp-matched chronological records, predicts
the current build's outcome from the immediately preceding build of the same
project. Only strictly earlier timestamps are accepted; timestamp ties and
projects without an earlier build are excluded from this baseline's scored set.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from features import TARGET_NAME, BUILD_ID

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FULL_FILE = DATA / "travistorrent_build_level_prebuild.csv"
CHRONO_FILE = DATA / "timestamp_matched_build_level.csv"
GROUP_RESULT_FILE = DATA / "project_grouped_evaluation.json"
OUTPUT_FILE = DATA / "baseline_comparison.json"
GROUP = "gh_project_name"
SEED = 42

def score(y_true, y_pred, name, note=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    actual_fail = (y_true == 0).astype(int)
    pred_fail = (y_pred == 0).astype(int)
    return {
        "baseline": name,
        "records_scored": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "failure_precision": float(precision_score(actual_fail, pred_fail, zero_division=0)),
        "failure_recall": float(recall_score(actual_fail, pred_fail, zero_division=0)),
        "failure_f1": float(f1_score(actual_fail, pred_fail, zero_division=0)),
        "failure_roc_auc": None,
        "confusion_matrix_labels_success_then_failure": confusion_matrix(
            y_true, y_pred, labels=[1, 0]
        ).tolist(),
        "note": note
    }

def majority_result(train, test, label):
    # Resolve ties deterministically to Success (1).
    counts = train[TARGET_NAME].astype(int).value_counts()
    majority = 1 if counts.get(1, 0) >= counts.get(0, 0) else 0
    pred = np.full(len(test), majority, dtype=int)
    result = score(test[TARGET_NAME], pred, "Majority-class baseline",
                   f"Always predicts {('Success' if majority == 1 else 'Failure')}, "
                   "selected using training labels only.")
    result["predicted_class"] = "Success" if majority == 1 else "Failure"
    result["experiment"] = label
    return result

def group_split(df):
    groups = df[GROUP].astype(str)
    y = df[TARGET_NAME].astype(int)
    all_idx = np.arange(len(df))
    global_failure_rate = float((y == 0).mean())
    best = None
    for seed in range(2000):
        outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        remain_pos, test_pos = next(outer.split(all_idx, y, groups))
        remain = df.iloc[remain_pos]
        inner = GroupShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=seed+10000)
        train_rel, val_rel = next(inner.split(np.arange(len(remain)), remain[TARGET_NAME], remain[GROUP].astype(str)))
        train_pos, val_pos = remain_pos[train_rel], remain_pos[val_rel]
        parts = [df.iloc[train_pos], df.iloc[val_pos], df.iloc[test_pos]]
        if any(p[TARGET_NAME].nunique() < 2 for p in parts):
            continue
        sizes = np.array([len(p) for p in parts], float)/len(df)
        rates = np.array([(p[TARGET_NAME].astype(int)==0).mean() for p in parts])
        score_value = np.abs(sizes - [0.70,0.15,0.15]).sum() + .35*np.abs(rates-global_failure_rate).sum()
        if best is None or score_value < best[0]:
            best = (score_value, train_pos, val_pos, test_pos)
    if best is None:
        raise RuntimeError("Could not reproduce project-grouped split.")
    return df.iloc[best[1]].copy(), df.iloc[best[2]].copy(), df.iloc[best[3]].copy()

def main():
    full = pd.read_csv(FULL_FILE, low_memory=False)
    full[TARGET_NAME] = pd.to_numeric(full[TARGET_NAME], errors="coerce")
    full = full.dropna(subset=[TARGET_NAME, GROUP])
    full = full[full[TARGET_NAME].isin([0,1])].reset_index(drop=True)

    train_r, temp = train_test_split(full, test_size=.30, random_state=SEED, stratify=full[TARGET_NAME].astype(int))
    val_r, test_r = train_test_split(temp, test_size=.50, random_state=SEED, stratify=temp[TARGET_NAME].astype(int))

    train_g, val_g, test_g = group_split(full)

    temporal = pd.read_csv(CHRONO_FILE, low_memory=False)
    temporal[TARGET_NAME] = pd.to_numeric(temporal[TARGET_NAME], errors="coerce")
    temporal["build_timestamp"] = pd.to_datetime(temporal["build_timestamp"], errors="coerce", utc=True)
    temporal = temporal.dropna(subset=[TARGET_NAME, "build_timestamp", GROUP])
    temporal = temporal[temporal[TARGET_NAME].isin([0,1])]
    temporal = temporal.sort_values(["build_timestamp", BUILD_ID]).reset_index(drop=True)
    n = len(temporal)
    i1, i2 = int(n*.70), int(n*.85)
    train_t, val_t, test_t = temporal.iloc[:i1], temporal.iloc[i1:i2], temporal.iloc[i2:]

    output = {
        "title": "Baseline comparison on held-out partitions",
        "majority_class_baseline": {
            "random": majority_result(train_r, test_r, "Random 70:15:15"),
            "chronological": majority_result(train_t, test_t, "Chronological 70:15:15"),
            "project_grouped": majority_result(train_g, test_g, "Project-grouped 70:15:15")
        },
        "previous_build_failure_baseline": {
            "availability": "Only evaluated on timestamp-matched chronological data where a strictly earlier prior build exists in the same project.",
            "chronological": None
        },
        "notes": [
            "Baseline metrics are calculated on the same test partitions as their corresponding Random Forest experiments.",
            "Majority class is selected from training labels only.",
            "The previous-build baseline uses the immediately preceding build outcome within the same project, ordered by build_timestamp.",
            "Rows with no earlier build in the same project or ambiguous equal timestamps are excluded from previous-build scoring.",
            "Baseline results are not used to fit or tune the Random Forest."
        ]
    }

    # Previous-build baseline: shift only when previous timestamp is strictly earlier.
    temporal["previous_build_success"] = temporal.groupby(GROUP)[TARGET_NAME].shift(1)
    temporal["previous_build_timestamp"] = temporal.groupby(GROUP)["build_timestamp"].shift(1)
    eligible = temporal[
        (temporal["build_timestamp"] > temporal["previous_build_timestamp"]) &
        temporal["previous_build_success"].notna()
    ].copy()
    # Restrict scored current records to the chronological held-out test window.
    test_ids = set(test_t[BUILD_ID].astype(str))
    eligible = eligible[eligible[BUILD_ID].astype(str).isin(test_ids)]
    if len(eligible):
        # Previous build success -> predict Success; previous build failure -> predict Failure.
        prev_pred = eligible["previous_build_success"].astype(int).to_numpy()
        prev_true = eligible[TARGET_NAME].astype(int).to_numpy()
        prev_result = score(
            prev_true, prev_pred, "Previous-build-outcome baseline",
            "Predicts the current build using the immediately previous build outcome in the same project."
        )
        prev_result["experiment"] = "Chronological held-out test"
        prev_result["test_records_available"] = int(len(test_t))
        prev_result["excluded_test_records"] = int(len(test_t)-len(eligible))
        prev_result["exclusion_reason"] = "No earlier same-project build or ambiguous timestamp tie."
        output["previous_build_failure_baseline"]["chronological"] = prev_result
    else:
        output["previous_build_failure_baseline"]["chronological"] = {
            "records_scored": 0, "note": "No eligible previous-build records found."
        }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Saved:", OUTPUT_FILE)
    print("MAJORITY CLASS BASELINE")
    for key, val in output["majority_class_baseline"].items():
        print(key, "predicted", val["predicted_class"], "records", val["records_scored"],
              "accuracy", round(val["accuracy"]*100,2), "%",
              "failure F1", round(val["failure_f1"]*100,2), "%")
    print("PREVIOUS BUILD BASELINE")
    print(json.dumps(output["previous_build_failure_baseline"]["chronological"], indent=2))

if __name__ == "__main__":
    main()
