import joblib
import shap
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "random_forest_model.joblib"

# Load model
model_data = joblib.load(MODEL_FILE)

model = model_data["model"]
FEATURES = model_data["features"]


def get_shap_explanation(input_data):

    # Convert input into DataFrame
    df = pd.DataFrame([input_data])

    # Keep exact training feature order
    df = df[FEATURES]

    # Get prediction
    prediction = int(model.predict(df)[0])

    # SHAP explainer
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(df)

    # Select SHAP values for the predicted class
    if isinstance(shap_values, list):

        values = shap_values[prediction][0]

    elif len(shap_values.shape) == 3:

        values = shap_values[0, :, prediction]

    else:

        values = shap_values[0]

    explanation = []

    for feature, value, shap_value in zip(
        FEATURES,
        df.iloc[0].values,
        values
    ):

        explanation.append({
            "feature": feature,
            "value": float(value),
            "shap_value": round(float(shap_value), 6),
            "absolute_shap": round(abs(float(shap_value)), 6)
        })

    # Highest influence first
    explanation.sort(
        key=lambda x: x["absolute_shap"],
        reverse=True
    )

    return explanation[:10]