# CI/CD Failure Prediction – 70:15:15 Evaluation Update

## What changed
- Added `backend/evaluate_70_15_15.py`.
- Added `GET /api/evaluation` to `backend/main.py`.
- Updated `backend/main.py` to apply training-set numeric medians saved in the model bundle.
- Updated `frontend/index.html` to fetch real held-out metrics and populate the comparison chart and confusion matrices.
- The model feature policy remains the exact 19 pre-build features in `backend/features.py`.

## Run the evaluation (from project root)
```bash
cd backend
python evaluate_70_15_15.py
```

This creates:
- `data/evaluation_70_15_15.json`
- updates `backend/random_forest_model.joblib`

Then start/restart FastAPI:
```bash
uvicorn main:app --reload
```
Open the frontend using your usual local web server. The evaluation panel fetches `/api/evaluation`.

## Split design
- Random: stratified 70% train / 15% validation / 15% held-out test across 18,095 build-level records.
- Chronological: timestamp-matched 4,704 records sorted by `build_timestamp`, earliest 70% train / next 15% validation / latest 15% held-out test.
- Model fitting uses only the training partition in each experiment. Validation and test are not used to fit.
- The FastAPI production prediction model is saved from the random experiment's training partition.
- Confusion matrix row/column order is Success, Failure.
- Failure precision/recall/F1 treat class 0 (Failure) as the positive class.

## Important research cautions
- The chronological experiment is restricted to the 4,704 timestamp-matched records.
- This is not project-grouped evaluation. The review also recommends a project-grouped split to test project-identity effects; that remains a separate future experiment.
- Do not compare these newly generated 70:15:15 values as if they were the earlier 80:20 experiment results. Report them as a new experiment.
- Do not tune model parameters against the held-out test set. If tuning is required, use validation only and rerun the final test once.
