import pandas as pd
import numpy as np
import shap

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_DIR / "train_ml.csv"
TEST_FILE = DATA_DIR / "test_ml.csv"

print("========== INDIVIDUAL BUILD EXPLANATION ==========")

# Load data
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Features and target
X_train = train_df.drop(columns=["tr_build_id", "build_success"])
y_train = train_df["build_success"]

X_test = test_df.drop(columns=["tr_build_id", "build_success"])
y_test = test_df["build_success"]

# Train model
print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed.")

# Select one build
build_index = 0

build_id = test_df.iloc[build_index]["tr_build_id"]
build_data = X_test.iloc[[build_index]]

actual_result = test_df.iloc[build_index]["build_success"]

# Prediction
prediction = model.predict(build_data)[0]
probability = model.predict_proba(build_data)[0]

print("\n========== BUILD INFORMATION ==========")
print("Build ID:", build_id)
print("Actual Result:", "SUCCESS" if actual_result == 1 else "FAILURE")
print("Predicted Result:", "SUCCESS" if prediction == 1 else "FAILURE")
print("Success Probability:", round(probability[1] * 100, 2), "%")
print("Failure Probability:", round(probability[0] * 100, 2), "%")

# SHAP
print("\nCreating SHAP explanation...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(build_data)

# Handle current SHAP format
if len(shap_values.shape) == 3:
    values = shap_values[0, :, 0]
else:
    values = shap_values[0]

explanation_df = pd.DataFrame({
    "feature": X_test.columns,
    "feature_value": build_data.iloc[0].values,
    "shap_value": values
})

# Sort by absolute SHAP contribution
explanation_df["absolute_shap"] = np.abs(
    explanation_df["shap_value"]
)

explanation_df = explanation_df.sort_values(
    by="absolute_shap",
    ascending=False
)

print("\n========== TOP SHAP FACTORS ==========")

print(
    explanation_df.head(10).to_string(index=False)
)

# Save explanation
OUTPUT_FILE = DATA_DIR / "build_explanation.csv"

explanation_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nExplanation saved to:")
print(OUTPUT_FILE)