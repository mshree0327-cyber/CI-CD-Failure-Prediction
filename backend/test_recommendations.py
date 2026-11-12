from recommendations import generate_recommendations

print("========== RECOMMENDATION ENGINE TEST ==========")

top_factors = [
    {
        "feature": "gh_sloc",
        "value": 1500,
        "shap_value": 0.05,
        "absolute_shap": 0.05
    },
    {
        "feature": "gh_team_size",
        "value": 2,
        "shap_value": 0.03,
        "absolute_shap": 0.03
    },
    {
        "feature": "git_diff_src_churn",
        "value": 150,
        "shap_value": 0.02,
        "absolute_shap": 0.02
    }
]

prediction = "FAILURE"

recommendations = generate_recommendations(
    prediction,
    top_factors
)

print("\nPrediction:", prediction)

print("\nRecommendations:")

for recommendation in recommendations:
    print("-", recommendation)