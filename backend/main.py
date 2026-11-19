"""FastAPI API aligned with the audited 19-feature pre-build model."""
from pathlib import Path
from typing import Optional
import joblib
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from  .recommendations import generate_recommendations

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "random_forest_model.joblib"
if not MODEL_FILE.exists():
    raise RuntimeError("Model file missing. Run train_leakage_aware_model.py first.")

bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
MODEL_FEATURES = bundle["features"]
RAW_FEATURES = bundle["raw_features"]
CAT_FEATURE = bundle["categorical_feature"]
NUMERIC_MEDIANS = bundle.get("numeric_medians", {})

# Load the newer 16-feature model independently; keep the original model above.
DYNAMIC_MODEL_FILE = BASE_DIR / "random_forest_dynamic_model.joblib"
if not DYNAMIC_MODEL_FILE.exists():
    raise RuntimeError(
        "Dynamic model file missing: backend/random_forest_dynamic_model.joblib"
    )

dynamic_bundle = joblib.load(DYNAMIC_MODEL_FILE)
dynamic_model = dynamic_bundle["model"]
DYNAMIC_FEATURES = dynamic_bundle["features"]
DYNAMIC_RAW_FEATURES = dynamic_bundle["raw_features"]
DYNAMIC_NUMERIC_MEDIANS = dynamic_bundle.get("numeric_medians", {})
DYNAMIC_CATEGORICAL_FEATURE = dynamic_bundle.get("categorical_feature")

if len(DYNAMIC_FEATURES) != 16 or DYNAMIC_FEATURES != DYNAMIC_RAW_FEATURES:
    raise RuntimeError("Dynamic model feature schema is not the expected 16 numeric features.")
if DYNAMIC_CATEGORICAL_FEATURE is not None:
    raise RuntimeError("Dynamic API expects the verified model with no categorical feature.")
EVALUATION_FILE = BASE_DIR.parent / "data" / "evaluation_70_15_15.json"
PROJECT_GROUPED_EVALUATION_FILE = BASE_DIR.parent / "data" / "project_grouped_evaluation.json"
BASELINE_COMPARISON_FILE = BASE_DIR.parent / "data" / "baseline_comparison.json"

app = FastAPI(
    title="Leakage-Aware CI/CD Failure Prediction API",
    description="Predicts build outcome using audited pre-build features only.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"]
)

class BuildData(BaseModel):
    git_diff_src_churn: float = Field(..., ge=0)
    gh_diff_files_added: float = Field(..., ge=0)
    gh_diff_files_deleted: float = Field(..., ge=0)
    gh_diff_files_modified: float = Field(..., ge=0)
    gh_diff_src_files: float = Field(..., ge=0)
    gh_diff_doc_files: float = Field(..., ge=0)
    gh_diff_other_files: float = Field(..., ge=0)
    gh_sloc: float = Field(..., ge=0)
    gh_num_commits_on_files_touched: float = Field(..., ge=0)
    code_churn_rate: float = Field(..., ge=0)
    files_changed_total: float = Field(..., ge=0)
    git_prev_commit_resolution_status: str
    gh_diff_tests_added: float = Field(..., ge=0)
    gh_diff_tests_deleted: float = Field(..., ge=0)
    gh_test_lines_per_kloc: float = Field(..., ge=0)
    gh_test_cases_per_kloc: float = Field(..., ge=0)
    gh_asserts_cases_per_kloc: float = Field(..., ge=0)
    gh_team_size: float = Field(..., ge=0)
    gh_by_core_team_member: float = Field(..., ge=0)


class DynamicBuildData(BaseModel):
    """The 16 numeric pre-build features used by the dynamic model."""
    git_diff_src_churn: float = Field(..., ge=0)
    gh_diff_files_added: float = Field(..., ge=0)
    gh_diff_files_deleted: float = Field(..., ge=0)
    gh_diff_files_modified: float = Field(..., ge=0)
    gh_diff_src_files: float = Field(..., ge=0)
    gh_diff_doc_files: float = Field(..., ge=0)
    gh_diff_other_files: float = Field(..., ge=0)
    gh_sloc: float = Field(..., ge=0)
    gh_num_commits_on_files_touched: float = Field(..., ge=0)
    code_churn_rate: float = Field(..., ge=0)
    files_changed_total: float = Field(..., ge=0)
    gh_diff_tests_added: float = Field(..., ge=0)
    gh_diff_tests_deleted: float = Field(..., ge=0)
    gh_test_lines_per_kloc: float = Field(..., ge=0)
    gh_test_cases_per_kloc: float = Field(..., ge=0)
    gh_asserts_cases_per_kloc: float = Field(..., ge=0)

def prepare_input(payload):
    raw = pd.DataFrame([payload])[RAW_FEATURES]
    raw[CAT_FEATURE] = raw[CAT_FEATURE].astype("string").fillna("__MISSING__")
    for column, median in NUMERIC_MEDIANS.items():
        raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(median)
    encoded = pd.get_dummies(raw, columns=[CAT_FEATURE], dtype=int)
    # Exact training columns and order; unseen categories become all-zero.
    return encoded.reindex(columns=MODEL_FEATURES, fill_value=0)


def prepare_dynamic_input(payload):
    """Build a one-row frame in the exact order used to train the 16-feature model."""
    raw = pd.DataFrame([payload]).reindex(columns=DYNAMIC_RAW_FEATURES)
    for column in DYNAMIC_RAW_FEATURES:
        median = DYNAMIC_NUMERIC_MEDIANS.get(column, 0.0)
        raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(median)
    return raw.reindex(columns=DYNAMIC_FEATURES)


def get_shap_values_for_class(model_to_explain, frame, predicted_class):
    """Return a one-dimensional SHAP vector across common SHAP output formats."""
    explainer = shap.TreeExplainer(model_to_explain)
    values = explainer.shap_values(frame)
    class_index = list(model_to_explain.classes_).index(predicted_class)

    if isinstance(values, list):
        selected = values[class_index]
    else:
        selected = values

    # SHAP Explanation objects and NumPy arrays both support conversion.
    import numpy as np
    selected = np.asarray(selected)
    if selected.ndim == 3:
        # Common multi-class layout: (samples, features, classes).
        selected = selected[0, :, class_index]
    elif selected.ndim == 2:
        selected = selected[0]
    elif selected.ndim != 1:
        raise ValueError(f"Unsupported SHAP output shape: {selected.shape}")

    return selected

@app.get("/")
def home():
    return {"message": "Leakage-aware CI/CD prediction API is running", "status": "success"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "legacy_model_loaded": model is not None,
        "legacy_feature_count": len(RAW_FEATURES),
        "dynamic_model_loaded": dynamic_model is not None,
        "dynamic_feature_count": len(DYNAMIC_RAW_FEATURES),
    }

@app.get("/api/features")
def get_features():
    return {"features": RAW_FEATURES, "count": len(RAW_FEATURES),
            "policy": "audited pre-build features only"}


@app.get("/api/features-dynamic")
def get_dynamic_features():
    return {
        "features": DYNAMIC_RAW_FEATURES,
        "count": len(DYNAMIC_RAW_FEATURES),
        "model": "random_forest_dynamic_model.joblib",
        "feature_policy": "16 numeric pre-build features",
        "historical_definition_parity": "unverified",
    }


@app.post("/api/predict-dynamic")
def predict_dynamic(build: DynamicBuildData):
    """Predict with the 16-feature model using pre-build inputs only."""
    try:
        frame = prepare_dynamic_input(build.dict())
        predicted = int(dynamic_model.predict(frame)[0])
        classes = list(dynamic_model.classes_)
        probabilities = dynamic_model.predict_proba(frame)[0]
        failure_probability = float(probabilities[classes.index(0)])
        success_probability = float(probabilities[classes.index(1)])
        result = "SUCCESS" if predicted == 1 else "FAILURE"

        top_factors = []
        explanation_status = "available"
        try:
            shap_vector = get_shap_values_for_class(dynamic_model, frame, predicted)
            factors = []
            for feature, value, impact in zip(DYNAMIC_FEATURES, frame.iloc[0], shap_vector):
                factors.append({
                    "feature": feature,
                    "value": float(value),
                    "shap_value": round(float(impact), 6),
                    "absolute_shap": round(abs(float(impact)), 6),
                })
            factors.sort(key=lambda item: item["absolute_shap"], reverse=True)
            top_factors = factors[:10]
        except Exception as shap_exc:
            # Prediction remains usable if the installed SHAP version returns an
            # unsupported format; report the explanation limitation explicitly.
            explanation_status = f"unavailable: {type(shap_exc).__name__}: {shap_exc}"

        try:
            recommendations = generate_recommendations(result, top_factors) if top_factors else []
        except Exception:
            recommendations = []

        return {
            "prediction": result,
            "failure_probability": round(failure_probability * 100, 2),
            "success_probability": round(success_probability * 100, 2),
            "model": "random_forest_dynamic_model.joblib",
            "feature_count": len(DYNAMIC_FEATURES),
            "explanation_status": explanation_status,
            "top_factors": top_factors,
            "recommendations": recommendations,
            "historical_definition_parity": "unverified",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Dynamic prediction failed: {exc}")


@app.post("/api/predict")
def predict(build: BuildData):
    try:
        frame = prepare_input(build.dict())
        predicted = int(model.predict(frame)[0])
        classes = list(model.classes_)
        probabilities = model.predict_proba(frame)[0]
        failure_probability = float(probabilities[classes.index(0)])
        success_probability = float(probabilities[classes.index(1)])
        result = "SUCCESS" if predicted == 1 else "FAILURE"

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(frame)
        if isinstance(shap_values, list):
            sv = shap_values[classes.index(predicted)][0]
        elif getattr(shap_values, "ndim", 0) == 3:
            sv = shap_values[0, :, classes.index(predicted)]
        else:
            sv = shap_values[0]

        factors = []
        for feature, value, impact in zip(MODEL_FEATURES, frame.iloc[0], sv):
            factors.append({
                "feature": feature, "value": float(value),
                "shap_value": round(float(impact), 6),
                "absolute_shap": round(abs(float(impact)), 6)
            })
        factors.sort(key=lambda item: item["absolute_shap"], reverse=True)
        top_factors = factors[:10]
        try:
            recommendations = generate_recommendations(result, top_factors)
        except Exception:
            recommendations = []
        return {
            "prediction": result,
            "failure_probability": round(failure_probability * 100, 2),
            "success_probability": round(success_probability * 100, 2),
            "top_factors": top_factors,
            "recommendations": recommendations
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")


@app.get("/api/evaluation")
def get_evaluation():
    """Return metrics computed by evaluate_70_15_15.py from held-out records."""
    if not EVALUATION_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation results not found. Run backend/evaluate_70_15_15.py first."
        )
    try:
        import json
        return json.loads(EVALUATION_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read evaluation results: {exc}")


@app.get("/api/project-grouped-evaluation")
def get_project_grouped_evaluation():
    """Return metrics from evaluation on projects excluded from training."""
    if not PROJECT_GROUPED_EVALUATION_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Project-grouped results not found. Run backend/evaluate_project_grouped.py first."
        )
    try:
        import json
        return json.loads(PROJECT_GROUPED_EVALUATION_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read project-grouped results: {exc}")


@app.get("/api/baseline-comparison")
def get_baseline_comparison():
    """Return majority-class and previous-build baseline metrics."""
    if not BASELINE_COMPARISON_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Baseline results not found. Run backend/evaluate_baselines.py first."
        )
    try:
        import json
        return json.loads(BASELINE_COMPARISON_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read baseline results: {exc}")


GLOBAL_SHAP_FILE = BASE_DIR.parent / "data" / "global_shap_importance.json"
GLOBAL_SHAP_SUMMARY = BASE_DIR.parent / "data" / "global_shap_summary.png"
GLOBAL_SHAP_BAR = BASE_DIR.parent / "data" / "global_shap_bar.png"

@app.get("/api/global-shap")
def get_global_shap():
    """Return global SHAP feature importance computed on held-out test samples."""
    if not GLOBAL_SHAP_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Global SHAP results not found. Run backend/global_shap_analysis.py first."
        )
    try:
        import json
        return json.loads(GLOBAL_SHAP_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read global SHAP results: {exc}")

@app.get("/api/global-shap/summary-plot")
def get_global_shap_summary_plot():
    if not GLOBAL_SHAP_SUMMARY.exists():
        raise HTTPException(status_code=404, detail="Run global_shap_analysis.py to generate the SHAP summary plot.")
    return FileResponse(GLOBAL_SHAP_SUMMARY, media_type="image/png")

@app.get("/api/global-shap/bar-plot")
def get_global_shap_bar_plot():
    if not GLOBAL_SHAP_BAR.exists():
        raise HTTPException(status_code=404, detail="Run global_shap_analysis.py to generate the SHAP bar plot.")
    return FileResponse(GLOBAL_SHAP_BAR, media_type="image/png")


ERROR_ANALYSIS_FILE = BASE_DIR.parent / "data" / "error_analysis.json"

@app.get("/api/error-analysis")
def get_error_analysis():
    """Return error counts and held-out test examples."""
    if not ERROR_ANALYSIS_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Error analysis results not found. Run backend/evaluate_error_analysis.py first."
        )
    try:
        import json
        return json.loads(ERROR_ANALYSIS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read error analysis results: {exc}")


ROC_PR_METRICS_FILE = BASE_DIR.parent / "data" / "roc_pr_curve_metrics.json"
ROC_CURVE_IMAGE = BASE_DIR.parent / "data" / "roc_curve.png"
PR_CURVE_IMAGE = BASE_DIR.parent / "data" / "precision_recall_curve.png"

@app.get("/api/evaluation-curves/metrics")
def get_evaluation_curve_metrics():
    if not ROC_PR_METRICS_FILE.exists():
        raise HTTPException(status_code=404, detail="Run backend/evaluate_roc_pr_curves.py first.")
    try:
        import json
        return json.loads(ROC_PR_METRICS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read ROC/PR metrics: {exc}")

@app.get("/api/evaluation-curves/roc")
def get_roc_curve_image():
    if not ROC_CURVE_IMAGE.exists():
        raise HTTPException(status_code=404, detail="ROC curve image not found. Run evaluate_roc_pr_curves.py.")
    return FileResponse(ROC_CURVE_IMAGE, media_type="image/png")

@app.get("/api/evaluation-curves/precision-recall")
def get_precision_recall_curve_image():
    if not PR_CURVE_IMAGE.exists():
        raise HTTPException(status_code=404, detail="Precision-Recall curve image not found. Run evaluate_roc_pr_curves.py.")
    return FileResponse(PR_CURVE_IMAGE, media_type="image/png")


BOOTSTRAP_CI_FILE = BASE_DIR.parent / "data" / "bootstrap_confidence_intervals.json"

@app.get("/api/bootstrap-confidence-intervals")
def get_bootstrap_confidence_intervals():
    if not BOOTSTRAP_CI_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Bootstrap results not found. Run backend/evaluate_bootstrap_ci.py first."
        )
    try:
        import json
        return json.loads(BOOTSTRAP_CI_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read bootstrap results: {exc}")


PROJECT_LEVEL_PERFORMANCE_FILE = BASE_DIR.parent / "data" / "project_level_performance.json"

@app.get("/api/project-level-performance")
def get_project_level_performance():
    if not PROJECT_LEVEL_PERFORMANCE_FILE.exists():
        raise HTTPException(status_code=404, detail="Project-level results not found. Run evaluate_project_level.py first.")
    try:
        import json
        return json.loads(PROJECT_LEVEL_PERFORMANCE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read project-level results: {exc}")


DECISION_THRESHOLD_FILE = BASE_DIR.parent / "data" / "decision_threshold_analysis.json"
DECISION_THRESHOLD_PLOT = BASE_DIR.parent / "data" / "decision_threshold_metrics.png"

@app.get("/api/decision-threshold-analysis")
def get_decision_threshold_analysis():
    if not DECISION_THRESHOLD_FILE.exists():
        raise HTTPException(status_code=404, detail="Run backend/evaluate_decision_threshold.py first.")
    try:
        import json
        return json.loads(DECISION_THRESHOLD_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read threshold results: {exc}")

@app.get("/api/decision-threshold-analysis/plot")
def get_decision_threshold_plot():
    if not DECISION_THRESHOLD_PLOT.exists():
        raise HTTPException(status_code=404, detail="Threshold plot not found. Run evaluate_decision_threshold.py.")
    return FileResponse(DECISION_THRESHOLD_PLOT, media_type="image/png")