from shap_api import get_shap_explanation

test_data = {
    "git_diff_src_churn": 0,
    "gh_diff_files_added": 0,
    "gh_diff_files_deleted": 0,
    "gh_diff_files_modified": 2,
    "gh_diff_src_files": 0,
    "gh_diff_doc_files": 0,
    "gh_diff_other_files": 0,
    "gh_sloc": 1110,
    "gh_num_commits_on_files_touched": 28,
    "code_churn_rate": 0,
    "files_changed_total": 2,
    "tr_build_number": 208,
    "git_prev_commit_resolution_status": 0,
    "git_diff_test_churn": 0,
    "gh_diff_tests_added": 0,
    "gh_diff_tests_deleted": 0,
    "gh_test_lines_per_kloc": 2854.054054,
    "gh_test_cases_per_kloc": 217.117117,
    "gh_asserts_cases_per_kloc": 275.675676,
    "gh_team_size": 5,
    "gh_by_core_team_member": 0,
    "gh_is_pr": 0
}

print("========== SHAP API TEST ==========")

results = get_shap_explanation(test_data)

for item in results:
    print(item)