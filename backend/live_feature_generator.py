"""
Generate the exact 16 numeric model inputs for a BASE..TARGET Git range.

Static metrics are calculated from the TARGET commit snapshot, not the
current working tree. Git diff metrics are calculated from BASE..TARGET.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from git_feature_extractor import extract_features as extract_git_features
from static_feature_extractor import extract_static_features

MODEL_FEATURES = [
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
    "gh_diff_tests_added",
    "gh_diff_tests_deleted",
    "gh_test_lines_per_kloc",
    "gh_test_cases_per_kloc",
    "gh_asserts_cases_per_kloc",
]


def _git_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError("Could not locate the Git repository root.")
    return Path(result.stdout.strip()).resolve()


def generate_features(base_commit: str, target_commit: str = "HEAD") -> dict:
    repo_root = _git_root()
    git_result = extract_git_features(base_commit, target_commit)
    static_result = extract_static_features(target_commit)

    git_values = git_result["extracted_features"]
    merged = {
        "git_diff_src_churn": git_values["git_diff_src_churn"],
        "gh_diff_files_added": git_values["gh_diff_files_added"],
        "gh_diff_files_deleted": git_values["gh_diff_files_deleted"],
        "gh_diff_files_modified": git_values["gh_diff_files_modified"],
        "gh_diff_src_files": git_values["gh_diff_src_files"],
        "gh_diff_doc_files": git_values["gh_diff_doc_files"],
        "gh_diff_other_files": git_values["gh_diff_other_files"],
        "gh_sloc": static_result["gh_sloc"],
        "gh_num_commits_on_files_touched": git_values["gh_num_commits_on_files_touched"],
        "code_churn_rate": git_values["code_churn_rate"],
        "files_changed_total": git_values["files_changed_total"],
        "gh_diff_tests_added": git_values["gh_diff_tests_added"],
        "gh_diff_tests_deleted": git_values["gh_diff_tests_deleted"],
        "gh_test_lines_per_kloc": static_result["gh_test_lines_per_kloc"],
        "gh_test_cases_per_kloc": static_result["gh_test_cases_per_kloc"],
        "gh_asserts_cases_per_kloc": static_result["gh_asserts_cases_per_kloc"],
    }

    missing = [name for name in MODEL_FEATURES if name not in merged]
    extra = [name for name in merged if name not in MODEL_FEATURES]
    if missing or extra or len(merged) != 16:
        raise ValueError(f"Feature schema mismatch: missing={missing}, extra={extra}")

    model_features = {}
    for name in MODEL_FEATURES:
        value = merged[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric; received {type(value).__name__}")
        model_features[name] = float(value)

    return {
        "repository_root": str(repo_root),
        "base_commit": base_commit,
        "target_commit": target_commit,
        "feature_count": len(MODEL_FEATURES),
        "feature_names": MODEL_FEATURES,
        "model_features": model_features,
        "git_diagnostics": git_result,
        "static_diagnostics": static_result,
        "validation_status": "SCHEMA_VALIDATED_DEFINITION_PARITY_UNVERIFIED",
        "warning": (
            "Static metrics now come from the target commit snapshot. "
            "They remain Python-only operational definitions and are not "
            "proven identical to the historical TravisTorrent calculations."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--target", default="HEAD")
    parser.add_argument("--output", default="build_features.json")
    args = parser.parse_args()

    result = generate_features(args.base, args.target)
    output_path = PROJECT_ROOT / args.output
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "feature_count": result["feature_count"],
        "feature_names": result["feature_names"],
        "validation_status": result["validation_status"],
        "static_target_commit": result["static_diagnostics"]["target_commit"],
        "static_source_files": result["static_diagnostics"]["python_source_files"],
        "static_test_files": result["static_diagnostics"]["python_test_files"],
        "static_test_cases": result["static_diagnostics"]["python_test_cases"],
        "static_assertions": result["static_diagnostics"]["python_assertions"],
        "warning": result["warning"],
    }, indent=2))
