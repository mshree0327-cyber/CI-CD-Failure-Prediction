BOOTSTRAP CONFIDENCE INTERVAL ANALYSIS UPDATE

What this adds:
- backend/evaluate_bootstrap_ci.py
- GET /api/bootstrap-confidence-intervals in backend/main.py
- Bootstrap Confidence Interval Analysis table in frontend/index.html
- data/bootstrap_confidence_intervals.json (sample generated output)

Method:
- Recreates the random stratified 70:15:15 split with random_state=42.
- Loads the existing backend/random_forest_model.joblib. It does NOT retrain.
- Uses only the held-out random test partition.
- Treats build_success=0 (Failure) as the positive class.
- Uses 2,000 non-parametric bootstrap resamples of paired test records.
- Reports percentile 95% confidence intervals (2.5th and 97.5th percentiles)
  for accuracy, failure precision, failure recall, failure F1 and ROC-AUC.
- The bootstrap seed is 2026 for reproducibility.

INSTALLATION:
1. Back up your current project.
2. Extract this ZIP into your project root and merge/replace the included backend
   and frontend files. Keep your existing dataset, model bundle and other files.
3. Confirm these existing files are present:
       data/travistorrent_build_level_prebuild.csv
       backend/random_forest_model.joblib
       backend/features.py
4. Activate your existing virtual environment.
5. From the backend directory run:
       python evaluate_bootstrap_ci.py
6. Restart FastAPI:
       uvicorn main:app --reload
7. Refresh the dashboard with Ctrl+F5.

The API endpoint is:
       http://127.0.0.1:8000/api/bootstrap-confidence-intervals

IMPORTANT:
The included JSON is a reproducible sample result from the dataset/model snapshot
used to prepare this patch. The results produced on your laptop after running
the script against your local dataset and model are the authoritative values
for your project report. If the local model/data differ, the confidence
intervals can differ. Do not copy the sample figures into the report without
checking your local script output.

Interpretation limitation:
These intervals quantify sampling uncertainty in the held-out test partition.
They do not account for uncertainty from retraining, model selection,
hyperparameter tuning or distribution shift.
