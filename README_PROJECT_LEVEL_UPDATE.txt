PROJECT-LEVEL PERFORMANCE ANALYSIS UPDATE

This adds per-repository subgroup evaluation for the existing Random Forest.

Files:
- backend/evaluate_project_level.py
- backend/main.py (GET /api/project-level-performance)
- frontend/index.html (Project-Level Performance Analysis table)
- data/project_level_performance.json (computed result)

Run from backend with the existing virtual environment activated:
    python evaluate_project_level.py

Then start/restart:
    uvicorn main:app --reload

Refresh the frontend using Ctrl+F5.

Method:
- Recreates the random stratified 70:15:15 split (random_state=42).
- Loads the existing production Random Forest model bundle.
- Predicts only the random held-out test partition.
- Groups those predictions by gh_project_name for descriptive subgroup metrics.
- gh_project_name is never passed to the model as an input feature.
- Flags projects with fewer than 10 test builds as small samples.

Important distinction:
This is per-project performance analysis on a random held-out test set. It is
NOT unseen-project generalization. The separate project-grouped evaluation
experiment is the one that holds out entire repositories from model training.
Small projects may have unstable precision/recall/F1, especially when they
contain very few failures.
