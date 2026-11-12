GLOBAL SHAP ANALYSIS UPDATE

What this adds:
- backend/global_shap_analysis.py
- Three FastAPI routes in backend/main.py:
    GET /api/global-shap
    GET /api/global-shap/summary-plot
    GET /api/global-shap/bar-plot
- Global SHAP chart, summary plot and feature-direction table in frontend/index.html

Research design:
- Loads the existing random_forest_model.joblib; it does not retrain the model.
- Recreates the same stratified random 70:15:15 split (random_state=42).
- Computes SHAP on a deterministic sample (up to 120 records) from the held-out test partition.
- Explains class label 0 (Failure).
- Aggregates one-hot encoded category columns back to the original 19 raw features.
- Positive mean signed SHAP means contribution towards Failure; negative means away from Failure.
- Mean absolute SHAP is contribution magnitude, not causality.

Install:
1. Back up your current project.
2. Extract this ZIP into your project root and merge/replace backend and frontend files.
3. Activate your existing virtual environment.
4. Ensure SHAP and matplotlib are installed:
       pip install shap matplotlib
   (SHAP is already listed in the project's requirements.txt.)
5. From the backend folder run:
       python global_shap_analysis.py
6. Confirm these files are created in data:
       global_shap_importance.json
       global_shap_summary.png
       global_shap_bar.png
7. Start/restart FastAPI:
       uvicorn main:app --reload
8. Refresh the frontend using Ctrl+F5.

API:
- http://127.0.0.1:8000/api/global-shap
- http://127.0.0.1:8000/api/global-shap/summary-plot
- http://127.0.0.1:8000/api/global-shap/bar-plot

If SHAP raises a package compatibility error, share the full terminal traceback before changing package versions.


DIRECTION ANALYSIS ENHANCEMENT:
The Global SHAP table now includes:
- Percentage of explained builds with positive SHAP contribution (towards Failure).
- Percentage with negative SHAP contribution (away from Failure).
- Percentage with zero contribution.
- Net direction based on mean signed SHAP.
Positive and negative percentages are calculated per original feature after
aggregating any one-hot encoded columns. These percentages may both be high
for a feature, showing that its effect varies across builds. Net direction
does not mean every build is affected in that direction.
