import pandas as pd
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
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

print("========== DECISION TREE MODEL ==========")

X_train = train_df.drop(columns=["tr_build_id", "build_success"])
y_train = train_df["build_success"]

X_test = test_df.drop(columns=["tr_build_id", "build_success"])
y_test = test_df["build_success"]

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, y_probability), 4)
)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))