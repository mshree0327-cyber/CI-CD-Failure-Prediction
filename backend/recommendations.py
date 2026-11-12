def generate_recommendations(prediction, top_factors):

    recommendations = []

    if prediction == "FAILURE":

        recommendations.append(
            "Review the recent code changes before rerunning the pipeline."
        )

    else:

        recommendations.append(
            "The build is currently predicted as successful. Continue with the pipeline."
        )

    for factor in top_factors:

        feature = factor["feature"]
        value = factor["value"]

        if feature == "gh_sloc" and value > 1000:
            recommendations.append(
                "The source code size is relatively high. Consider reviewing large or complex changes."
            )

        elif feature == "files_changed_total" and value > 10:
            recommendations.append(
                "A large number of files were changed. Review the modified files carefully."
            )

        elif feature == "git_diff_src_churn" and value > 100:
            recommendations.append(
                "High source-code churn was detected. Review additions and deletions before the build."
            )

        elif feature == "gh_team_size" and value <= 2:
            recommendations.append(
                "The project has a small team size. Consider additional code review before merging."
            )

        elif feature == "gh_test_lines_per_kloc" and value < 100:
            recommendations.append(
                "Test-code density is relatively low. Consider increasing automated test coverage."
            )

    # Remove duplicate recommendations
    recommendations = list(dict.fromkeys(recommendations))

    return recommendations[:5]