BASELINE COMPARISON UPDATE

This patch adds two simple baseline methods for research comparison.

1. Majority-class baseline:
   Always predicts the most frequent outcome in the TRAINING partition.
   It is evaluated on the same held-out test records for:
   - Random 70:15:15
   - Chronological 70:15:15
   - Project-grouped 70:15:15

2. Previous-build-outcome baseline:
   Uses the immediately preceding build outcome from the same project,
   ordered by build_timestamp, to predict the current build outcome.
   It is evaluated only on the chronological timestamp-matched held-out
   test records. Rows without a strictly earlier previous build, including
   ambiguous timestamp ties, are excluded and counted.

Install:
- Extract this patch into the project root and merge backend, frontend, data.
- Activate the existing venv.
- From backend run: python evaluate_baselines.py
- Start API: uvicorn main:app --reload
- Open frontend using Live Server and refresh.

API:
GET http://127.0.0.1:8000/api/baseline-comparison

Research caution:
The previous-build baseline is a simple persistence baseline. Its result
is a useful comparison, not evidence that the Random Forest is superior.
The baseline must not be used to tune the Random Forest on the held-out test.
