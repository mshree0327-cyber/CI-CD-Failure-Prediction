import pandas as pd
import numpy as np
import shap

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_DIR / "train_ml.csv"
TEST_FILE = DATA_DIR / "test_ml.csv"

print("========== RANDOM FOREST + SHAP ==========")

# Load data
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Separate features and target
X_train = train_df.drop(columns=["tr_build_id", "build_success"])
y_train = train_df["build_success"]

X_test = test_df.drop(columns=["tr_build_id", "build_success"])
y_test = test_df["build_success"]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Features:", len(X_train.columns))

# Train Random Forest
print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Random Forest training completed.")

# Prediction
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# SHAP explainer
print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(model)

# Explain a small sample first
X_sample = X_test.iloc[:100]

shap_values = explainer.shap_values(X_sample)

print("SHAP explanation generated.")

print("\nSHAP values shape:")

if isinstance(shap_values, list):
    for i, values in enumerate(shap_values):
        print("Class", i, ":", values.shape)
else:
    print(shap_values.shape)

# Global feature importance
print("\n========== GLOBAL SHAP FEATURE IMPORTANCE ==========")

# SHAP output format:
# (samples, features, classes)
# We use class 1 = build success/failure target class as returned by the model.

if len(shap_values.shape) == 3:
    # Select class 1
    values = shap_values[:, :, 1]
else:
    values = shap_values

mean_abs_shap = np.abs(values).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": X_sample.columns,
    "mean_abs_shap": mean_abs_shap
})

importance_df = importance_df.sort_values(
    by="mean_abs_shap",
    ascending=False
)

print("\nTop features:")

print(
    importance_df.head(10).to_string(index=False)
)

# Save feature importance
OUTPUT_FILE = DATA_DIR / "shap_feature_importance.csv"

importance_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSHAP feature importance saved to:")
print(OUTPUT_FILE)