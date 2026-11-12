DECISION THRESHOLD ANALYSIS UPDATE

Adds:
- backend/evaluate_decision_threshold.py
- GET /api/decision-threshold-analysis
- GET /api/decision-threshold-analysis/plot
- Decision Threshold Analysis dashboard section
- data/decision_threshold_analysis.json
- data/decision_threshold_metrics.png

Method:
- Recreates the random stratified 70:15:15 split (random_state=42).
- Loads the existing Random Forest bundle; does not retrain it.
- Uses failure probability for build_success=0 as the positive class.
- Compares thresholds from 0.10 to 0.90 in increments of 0.05.
- Selects the threshold on VALIDATION data only, maximizing Failure F1.
- If multiple thresholds tie on validation F1, selects the higher threshold.
- Evaluates the selected threshold once on the held-out TEST set.
- Also reports test performance at the default 0.50 threshold for comparison.

INSTALL:
1. Back up your current project.
2. Extract this ZIP into the project root and merge/replace backend/main.py and
   frontend/index.html. Keep your existing dataset and model bundle.
3. Confirm these existing files are present:
   data/travistorrent_build_level_prebuild.csv
   backend/random_forest_model.joblib
   backend/features.py
4. Activate your venv and run from backend:
   python evaluate_decision_threshold.py
5. Restart FastAPI:
   uvicorn main:app --reload
6. Refresh dashboard with Ctrl+F5.

Important:
- Do not select the threshold using held-out test metrics.
- The included JSON/plot are generated from the project archive snapshot. Your
  laptop's output is authoritative if local dataset/model versions differ.
- The validation-F1-optimal threshold is a research selection rule, not a
  universal operational optimum. Production deployment should consider the
  business costs of missed failures and false alarms.
