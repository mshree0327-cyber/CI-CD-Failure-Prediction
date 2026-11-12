import pandas as pd
from pathlib import Path

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

train_df = pd.read_csv(DATA_DIR / "train_ml.csv")
test_df = pd.read_csv(DATA_DIR / "test_ml.csv")

print("========== SVM MODEL ==========")

X_train = train_df.drop(columns=["tr_build_id", "build_success"])
y_train = train_df["build_success"]

X_test = test_df.drop(columns=["tr_build_id", "build_success"])
y_test = test_df["build_success"]

print("Scaling data...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training SVM...")

model = SVC(
    kernel="rbf",
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("SVM training completed.")

y_pred = model.predict(X_test_scaled)
y_score = model.decision_function(X_test_scaled)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, y_score), 4)
)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))