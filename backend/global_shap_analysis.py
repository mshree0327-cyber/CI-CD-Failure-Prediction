"""Compute global SHAP importance for the existing 19-feature Random Forest.

Uses the exact random 70:15:15 split and explains a deterministic stratified
sample from its held-out test partition. It does not retrain or tune the model.
SHAP values explain the Failure class (target label 0).
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from features import PREBUILD_FEATURES, TARGET_NAME

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "travistorrent_build_level_prebuild.csv"
MODEL_FILE = Path(__file__).resolve().parent / "random_forest_model.joblib"
JSON_OUT = ROOT / "data" / "global_shap_importance.json"
SUMMARY_PNG = ROOT / "data" / "global_shap_summary.png"
BAR_PNG = ROOT / "data" / "global_shap_bar.png"
SEED = 42
MAX_EXPLAIN_ROWS = 120
PLOT_ROWS = 120

def main():
    if not DATA_FILE.exists() or not MODEL_FILE.exists():
        raise FileNotFoundError("Dataset or random_forest_model.joblib is missing.")
    bundle = joblib.load(MODEL_FILE)
    if len(bundle.get("raw_features", [])) != 19:
        raise ValueError("Expected the trained 19-feature model bundle.")

    df = pd.read_csv(DATA_FILE, low_memory=False)
    df[TARGET_NAME] = pd.to_numeric(df[TARGET_NAME], errors="coerce")
    df = df.dropna(subset=[TARGET_NAME])
    df = df[df[TARGET_NAME].isin([0, 1])].reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=SEED,
        stratify=df[TARGET_NAME].astype(int)
    )
    _, test_df = train_test_split(
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

    # Deterministic stratified sample from held-out test only.
    y = test_df[TARGET_NAME].astype(int)
    rng = np.random.default_rng(SEED)
    selected = []
    for label in (0, 1):
        idx = y[y == label].index.to_numpy()
        n = min(int(round(MAX_EXPLAIN_ROWS * len(idx) / len(y))), len(idx))
        if n:
            selected.extend(rng.choice(idx, n, replace=False).tolist())
    if len(selected) < min(MAX_EXPLAIN_ROWS, len(test_df)):
        remaining = np.setdiff1d(test_df.index.to_numpy(), np.array(selected, dtype=int))
        selected.extend(rng.choice(remaining, min(MAX_EXPLAIN_ROWS-len(selected), len(remaining)), replace=False).tolist())
    selected = np.array(sorted(selected[:MAX_EXPLAIN_ROWS]), dtype=int)
    Xs = X.loc[selected]

    explainer = shap.TreeExplainer(bundle["model"])
    raw_values = explainer.shap_values(Xs)
    class_index = list(bundle["model"].classes_).index(0)  # Failure
    if isinstance(raw_values, list):
        sv = np.asarray(raw_values[class_index])
    else:
        raw_values = np.asarray(raw_values)
        sv = raw_values[:, :, class_index] if raw_values.ndim == 3 else raw_values
    if sv.shape != Xs.shape:
        raise ValueError(f"Unexpected SHAP dimensions {sv.shape}; input shape {Xs.shape}")

    encoded_names = list(Xs.columns)
    feature_rows = []
    for raw_feature in PREBUILD_FEATURES:
        cols = ([raw_feature] if raw_feature != cat else
                [c for c in encoded_names if c == cat or c.startswith(cat + "_")])
        indices = [encoded_names.index(c) for c in cols if c in encoded_names]
        if not indices:
            continue
        per_row = sv[:, indices].sum(axis=1)
        positive_values = per_row[per_row > 0]
        negative_values = per_row[per_row < 0]
        mean_signed = float(per_row.mean())
        if mean_signed > 1e-12:
            net_direction = "Net towards Failure"
        elif mean_signed < -1e-12:
            net_direction = "Net away from Failure"
        else:
            net_direction = "Approximately neutral"

        feature_rows.append({
            "feature": raw_feature,
            "mean_abs_shap": float(np.abs(sv[:, indices]).sum(axis=1).mean()),
            "mean_signed_shap": mean_signed,
            "positive_contribution_pct": float((per_row > 0).mean() * 100.0),
            "negative_contribution_pct": float((per_row < 0).mean() * 100.0),
            "zero_contribution_pct": float((per_row == 0).mean() * 100.0),
            "mean_positive_shap_when_positive": (
                float(positive_values.mean()) if len(positive_values) else 0.0
            ),
            "mean_negative_shap_when_negative": (
                float(negative_values.mean()) if len(negative_values) else 0.0
            ),
            "net_direction": net_direction
        })
    feature_rows.sort(key=lambda r: r["mean_abs_shap"], reverse=True)

    result = {
        "title": "Global SHAP feature importance",
        "explained_class": "Failure",
        "method": "SHAP TreeExplainer",
        "model": "Existing Random Forest trained on random 70% training partition",
        "source_partition": "Random stratified held-out test partition",
        "test_partition_records": int(len(test_df)),
        "explained_sample_records": int(len(Xs)),
        "raw_feature_count": 19,
        "encoded_feature_count": int(Xs.shape[1]),
        "aggregation": "One-hot encoded columns are aggregated to their original raw feature.",
        "features": feature_rows,
        "summary_plot_url": "/api/global-shap/summary-plot",
        "bar_plot_url": "/api/global-shap/bar-plot",
        "notes": [
            "The model is not retrained by this script.",
            "SHAP is calculated on a deterministic sample from the held-out test partition.",
            "Positive SHAP contributions move the model output toward the Failure class; negative contributions move it away.",
            "Positive and negative contribution percentages are calculated across explained builds for each original feature.",
            "Mean positive and mean negative SHAP values are conditional averages over builds with contributions of that sign.",
            "Net direction is based on the mean signed SHAP value and does not mean every build has that direction.",
            "Mean absolute SHAP is an average contribution magnitude, not a causal effect."
        ]
    }
    JSON_OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    shap.summary_plot(sv, Xs, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(SUMMARY_PNG, dpi=160, bbox_inches="tight")
    plt.close()

    top = feature_rows[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh([r["feature"] for r in top], [r["mean_abs_shap"] for r in top])
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Global SHAP importance – Failure class")
    plt.tight_layout()
    plt.savefig(BAR_PNG, dpi=160, bbox_inches="tight")
    plt.close()

    print("Global SHAP analysis completed.")
    print("Held-out test records:", len(test_df), "| explained sample:", len(Xs))
    print("Saved:", JSON_OUT)
    print("Saved:", SUMMARY_PNG)
    print("Saved:", BAR_PNG)
    print("Top 10 features:")
    for item in feature_rows[:10]:
        print(
    f'{item["feature"]}: '
    f'mean_abs={item["mean_abs_shap"]:.6f}, '
    f'mean_signed={item["mean_signed_shap"]:.6f}, '
    f'towards_failure={item["positive_contribution_pct"]:.2f}%, '
    f'away_from_failure={item["negative_contribution_pct"]:.2f}%, '
    f'direction={item["net_direction"]}'
)

if __name__ == "__main__":
    main()
