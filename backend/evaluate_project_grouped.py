"""Project-grouped 70:15:15 evaluation.

Entire repositories (gh_project_name) are assigned to exactly one partition.
No repository appears in more than one partition. This tests generalization
to unseen projects. It does not replace random or chronological evaluation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import GroupShuffleSplit
from features import PREBUILD_FEATURES, TARGET_NAME, BUILD_ID

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "travistorrent_build_level_prebuild.csv"
OUTPUT_FILE = ROOT / "data" / "project_grouped_evaluation.json"
GROUP_COLUMN = "gh_project_name"
SEED = 42

def make_group_splits(df):
    """Search deterministic GroupShuffleSplit candidates for near 70/15/15 rows.

    GroupShuffleSplit splits groups, not rows, so exact row percentages are
    generally impossible when project sizes differ. Select the candidate
    closest to desired row counts and class proportions, while preserving
    strict group separation and both classes in every partition.
    """
    groups = df[GROUP_COLUMN].astype(str)
    y = df[TARGET_NAME].astype(int)
    all_idx = np.arange(len(df))
    global_failure_rate = float((y == 0).mean())
    best = None

    for seed in range(2000):
        outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        remain_pos, test_pos = next(outer.split(all_idx, y, groups))
        remain_df = df.iloc[remain_pos]
        remain_groups = remain_df[GROUP_COLUMN].astype(str)
        remain_y = remain_df[TARGET_NAME].astype(int)

        inner = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85,
                                  random_state=seed + 10000)
        train_rel, val_rel = next(inner.split(
            np.arange(len(remain_df)), remain_y, remain_groups
        ))
        train_pos = remain_pos[train_rel]
        val_pos = remain_pos[val_rel]

        parts = [df.iloc[train_pos], df.iloc[val_pos], df.iloc[test_pos]]
        if any(len(p) == 0 or p[TARGET_NAME].nunique() < 2 for p in parts):
            continue

        # Penalize deviations from target row proportions and class balance.
        target_sizes = np.array([0.70, 0.15, 0.15])
        actual_sizes = np.array([len(p) for p in parts], dtype=float) / len(df)
        size_error = np.abs(actual_sizes - target_sizes).sum()
        rates = np.array([(p[TARGET_NAME].astype(int) == 0).mean() for p in parts])
        rate_error = np.abs(rates - global_failure_rate).sum()
        score = size_error + 0.35 * rate_error

        # Ensure project groups are disjoint.
        project_sets = [set(p[GROUP_COLUMN].astype(str)) for p in parts]
        if (project_sets[0] & project_sets[1] or
            project_sets[0] & project_sets[2] or
            project_sets[1] & project_sets[2]):
            continue

        if best is None or score < best[0]:
            best = (score, train_pos, val_pos, test_pos, seed)

    if best is None:
        raise RuntimeError("Could not create valid group-disjoint train/validation/test partitions.")
    return best[1], best[2], best[3], best[4]

def prepare_features(train_df, val_df, test_df):
    cat = "git_prev_commit_resolution_status"
    numeric = [c for c in PREBUILD_FEATURES if c != cat]
    frames = [p[PREBUILD_FEATURES].copy() for p in (train_df, val_df, test_df)]
    medians = {}
    for col in numeric:
        for frame in frames:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        med = frames[0][col].median()
        medians[col] = float(med) if pd.notna(med) else 0.0
        for frame in frames:
            frame[col] = frame[col].fillna(medians[col])
    for frame in frames:
        frame[cat] = frame[cat].astype("string").fillna("__MISSING__").astype(str)

    train_x = pd.get_dummies(frames[0], columns=[cat], dtype=int)
    val_x = pd.get_dummies(frames[1], columns=[cat], dtype=int).reindex(
        columns=train_x.columns, fill_value=0
    )
    test_x = pd.get_dummies(frames[2], columns=[cat], dtype=int).reindex(
        columns=train_x.columns, fill_value=0
    )
    return train_x, val_x, test_x

def metrics(model, X, y, partition):
    pred = model.predict(X)
    prob = model.predict_proba(X)
    classes = list(model.classes_)
    fail_prob = prob[:, classes.index(0)]
    actual_fail = (y.astype(int) == 0).astype(int)
    pred_fail = (pred == 0).astype(int)
    auc = float(roc_auc_score(actual_fail, fail_prob)) if y.nunique() == 2 else None
    return {
        "partition": partition,
        "records": int(len(y)),
        "projects": int(X.shape[0]) if False else None,
        "class_distribution": {
            "success": int((y.astype(int) == 1).sum()),
            "failure": int((y.astype(int) == 0).sum())
        },
        "accuracy": float(accuracy_score(y, pred)),
        "failure_precision": float(precision_score(actual_fail, pred_fail, zero_division=0)),
        "failure_recall": float(recall_score(actual_fail, pred_fail, zero_division=0)),
        "failure_f1": float(f1_score(actual_fail, pred_fail, zero_division=0)),
        "failure_roc_auc": auc,
        "confusion_matrix_labels_success_then_failure": confusion_matrix(
            y.astype(int), pred.astype(int), labels=[1, 0]
        ).tolist(),
        "classification_report": classification_report(
            y, pred, labels=[0, 1], target_names=["Failure", "Success"],
            output_dict=True, zero_division=0
        )
    }

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    required = PREBUILD_FEATURES + [TARGET_NAME, GROUP_COLUMN, BUILD_ID]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df[TARGET_NAME] = pd.to_numeric(df[TARGET_NAME], errors="coerce")
    df = df.dropna(subset=[TARGET_NAME, GROUP_COLUMN])
    df = df[df[TARGET_NAME].isin([0, 1])].reset_index(drop=True)

    train_idx, val_idx, test_idx, split_seed = make_group_splits(df)
    train_df, val_df, test_df = df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()

    X_train, X_val, X_test = prepare_features(train_df, val_df, test_df)
    y_train = train_df[TARGET_NAME].astype(int)
    y_val = val_df[TARGET_NAME].astype(int)
    y_test = test_df[TARGET_NAME].astype(int)

    model = RandomForestClassifier(
        n_estimators=150, random_state=SEED, class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)

    def partition_info(part):
        return {
            "records": int(len(part)),
            "projects": int(part[GROUP_COLUMN].nunique()),
            "project_names": sorted(part[GROUP_COLUMN].astype(str).unique().tolist()),
            "class_distribution": {
                "success": int((part[TARGET_NAME].astype(int) == 1).sum()),
                "failure": int((part[TARGET_NAME].astype(int) == 0).sum())
            }
        }

    output = {
        "title": "Project-grouped 70:15:15 evaluation",
        "group_column": GROUP_COLUMN,
        "target": TARGET_NAME,
        "feature_count_before_encoding": len(PREBUILD_FEATURES),
        "features": PREBUILD_FEATURES,
        "split_policy": (
            "Project-disjoint GroupShuffleSplit. Every gh_project_name is assigned "
            "to exactly one of train, validation, or held-out test. Percentages are "
            "approximate by record count because project sizes vary."
        ),
        "random_state_for_split_search": int(split_seed),
        "model": "RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)",
        "train": partition_info(train_df),
        "validation": partition_info(val_df),
        "test": partition_info(test_df),
        "validation_metrics": metrics(model, X_val, y_val, "validation"),
        "test_metrics": metrics(model, X_test, y_test, "held_out_unseen_projects"),
        "checks": {
            "train_validation_project_overlap": sorted(set(train_df[GROUP_COLUMN]) & set(val_df[GROUP_COLUMN])),
            "train_test_project_overlap": sorted(set(train_df[GROUP_COLUMN]) & set(test_df[GROUP_COLUMN])),
            "validation_test_project_overlap": sorted(set(val_df[GROUP_COLUMN]) & set(test_df[GROUP_COLUMN])),
            "all_projects_assigned_once": (
                train_df[GROUP_COLUMN].nunique() +
                val_df[GROUP_COLUMN].nunique() +
                test_df[GROUP_COLUMN].nunique() == df[GROUP_COLUMN].nunique()
            )
        },
        "notes": [
            "gh_project_name is used only to assign groups and is excluded from model inputs.",
            "The model is fitted only on the training projects.",
            "Validation and held-out test projects are not used for fitting.",
            "This is a separate experiment from random and chronological evaluation.",
            "This evaluates generalization to unseen projects; it does not establish chronological generalization."
        ]
    }
    OUTPUT_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Saved:", OUTPUT_FILE)
    for label, part in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"{label}: {len(part)} records, {part[GROUP_COLUMN].nunique()} projects, "
              f"{(part[TARGET_NAME] == 0).sum()} failures, {(part[TARGET_NAME] == 1).sum()} successes")
    print("Test metrics:")
    for key in ["accuracy", "failure_precision", "failure_recall", "failure_f1", "failure_roc_auc"]:
        value = output["test_metrics"][key]
        print(f"  {key}: {value:.4f}" if value is not None else f"  {key}: unavailable")
    print("Project overlap checks:", output["checks"])

if __name__ == "__main__":
    main()
