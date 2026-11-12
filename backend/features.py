
"""Leakage-aware feature policy for TravisTorrent."""

TARGET_NAME = "build_success"
BUILD_ID = "tr_build_id"

PREBUILD_FEATURES = [
    "git_diff_src_churn",
    "gh_diff_files_added",
    "gh_diff_files_deleted",
    "gh_diff_files_modified",
    "gh_diff_src_files",
    "gh_diff_doc_files",
    "gh_diff_other_files",
    "gh_sloc",
    "gh_num_commits_on_files_touched",
    "code_churn_rate",
    "files_changed_total",
    "git_prev_commit_resolution_status",
    "gh_diff_tests_added",
    "gh_diff_tests_deleted",
    "gh_test_lines_per_kloc",
    "gh_test_cases_per_kloc",
    "gh_asserts_cases_per_kloc",
    "gh_team_size",
    "gh_by_core_team_member",
]

DIRECT_OUTCOME_COLUMNS = [
    "tr_status",
    "build_success",
    "build_success.1",
]

EXECUTION_DEPENDENT_COLUMNS = [
    "tr_duration",
    "tr_log_buildduration",
    "tr_log_setup_time",
    "build_time_minutes",
    "setup_time_minutes",
    "build_efficiency",
    "tr_log_bool_tests_ran",
    "tr_log_bool_tests_failed",
    "tr_log_num_tests_ok",
    "tr_log_num_tests_failed",
    "tr_log_num_tests_run",
    "tr_log_num_tests_skipped",
    "tr_log_num_test_suites_run",
    "tr_log_num_test_suites_ok",
    "tr_log_num_test_suites_failed",
    "tr_log_testduration",
    "test_pass_rate",
    "test_coverage_proxy",
    "test_density",
    "test_suite_success_rate",
    "test_execution_time_min",
    "test_success_rate",
    "quality_score",
    "build_duration_minutes",
]

FUTURE_OR_IDENTITY_RISK_COLUMNS = [
    "rolling_avg_duration_3",
    "rolling_avg_duration_5",
    "tr_build_number",
    "build_hour",
    "build_day_of_week",
    "build_month",
    "build_year",
    "build_is_weekend",
]

METADATA_COLUMNS = [
    "tr_build_id",
    "gh_project_name",
]

EXCLUSION_REASONS = {
    **{
        c: "Direct current-build outcome or duplicate target"
        for c in DIRECT_OUTCOME_COLUMNS
    },
    **{
        c: "Execution-dependent feature"
        for c in EXECUTION_DEPENDENT_COLUMNS
    },
    **{
        c: "Potential temporal or identity leakage"
        for c in FUTURE_OR_IDENTITY_RISK_COLUMNS
    },
}
