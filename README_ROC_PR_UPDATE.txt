ROC AND PRECISION-RECALL CURVE UPDATE

This update adds:
- backend/evaluate_roc_pr_curves.py
- FastAPI endpoints:
    GET /api/evaluation-curves/metrics
    GET /api/evaluation-curves/roc
    GET /api/evaluation-curves/precision-recall
- A dashboard section displaying ROC and Precision-Recall plots and metrics.
- Generated files:
    data/roc_pr_curve_metrics.json
    data/roc_curve.png
    data/precision_recall_curve.png

INSTALLATION:
1. Back up your current project.
2. Extract this ZIP into your project root and merge/replace the backend and
   frontend files. This package's index.html retains the previous Global SHAP,
   project-level and error-analysis dashboard sections from the prior update.
3. Activate the existing virtual environment.
4. From the backend directory run:
       python evaluate_roc_pr_curves.py
5. Restart FastAPI:
       uvicorn main:app --reload
6. Refresh the dashboard using Ctrl+F5.

METHOD:
- Recreates the same random stratified 70:15:15 split (random_state=42).
- Loads the existing Random Forest bundle; it does not retrain the model.
- Scores only the 2,715-record held-out random test partition.
- Treats build_success=0 (Failure) as the positive class.
- ROC-AUC is computed from failure probabilities.
- Average Precision is reported for the precision-recall curve.
- The PR chart includes failure prevalence as its reference line.

IMPORTANT:
These curves summarize performance across thresholds. They do not choose a
decision threshold. Threshold selection and confidence intervals are separate
follow-up analyses. The generated metrics in this ZIP were calculated using
the included model and dataset snapshot; rerun on your own project to confirm
the same results in your environment.
