import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TRAIN_FILE = DATA_DIR / "train_ml.csv"
MODEL_FILE = BASE_DIR / "backend" / "random_forest_model.joblib"

print("========== SAVING RANDOM FOREST MODEL ==========")

train_df = pd.read_csv(TRAIN_FILE)

X_train = train_df.drop(columns=["tr_build_id", "build_success"])
y_train = train_df["build_success"]

print("Training rows:", len(X_train))
print("Features:", len(X_train.columns))

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

joblib.dump(
    {
        "model": model,
        "features": list(X_train.columns)
    },
    MODEL_FILE
)

print("\nModel saved successfully!")
print("Model file:")
print(MODEL_FILE)

print("\nFeatures saved:")
for feature in X_train.columns:
    print("-", feature)