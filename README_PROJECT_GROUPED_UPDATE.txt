PROJECT-GROUPED EVALUATION UPDATE

This patch adds a separate unseen-project evaluation experiment.

Files:
- backend/evaluate_project_grouped.py
- backend/main.py (adds GET /api/project-grouped-evaluation)
- frontend/index.html (adds Project-Grouped Evaluation card)
- data/project_grouped_evaluation.json (computed result)

Installation:
1. Back up your current project.
2. Extract this ZIP into your project root.
3. Allow the backend, frontend and data folders to merge/replace files.
4. Activate your existing virtual environment.
5. From the backend folder run:
   python evaluate_project_grouped.py
6. Start the API:
   uvicorn main:app --reload
7. Open the frontend with Live Server and refresh.

The evaluation script uses gh_project_name only to assign whole projects
to train, validation and held-out test partitions. It is not a model feature.
The split is approximately 70:15:15 by build-record count because projects
have different numbers of builds. It checks that no project overlaps
between partitions.

This project-grouped experiment is separate from the random and
chronological experiments. Its low failure recall should be discussed
honestly as a limitation, not hidden or replaced with the other results.
