
import argparse
import json
import subprocess
import sys
from pathlib import Path

from git_feature_extractor import classify_file


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

GIT_EXTRACTOR = BACKEND_DIR / "git_feature_extractor.py"
STATIC_EXTRACTOR = BACKEND_DIR / "static_feature_extractor.py"

OUTPUT_FILE = PROJECT_ROOT / "build_features.json"


# =========================================================
# COMMAND EXECUTION
# =========================================================

def run_command(command):
    """Execute a command from the project root."""

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)

    return result.stdout


# =========================================================
# TARGET SOURCE LOC
# =========================================================

def get_target_source_loc(target_commit):
    """
    Count physical lines in tracked source files
    at the target Git revision.

    Test files and non-source files are excluded.
    Blank lines and comments are included.
    """

    output = run_command([
        "git",
        "ls-tree",
        "-r",
        "--name-only",
        target_commit
    ])

    total_lines = 0

    for filename in output.splitlines():

        if classify_file(filename) != "source":
            continue

        file_content = run_command([
            "git",
            "show",
            f"{target_commit}:{filename}"
        ])

        total_lines += len(file_content.splitlines())

    return total_lines


# =========================================================
# COMMIT HISTORY
# =========================================================

def count_commits_on_touched_files(
    base_commit,
    target_commit,
    changed_files
):
    """
    Sum the number of commits affecting each changed file
    within the selected comparison range.
    """

    total_commits = 0

    for filename in changed_files:

        output = run_command([
            "git",
            "log",
            "--format=%H",
            f"{base_commit}..{target_commit}",
            "--",
            filename
        ])

        commits = [
            line for line in output.splitlines()
            if line.strip()
        ]

        total_commits += len(commits)

    return total_commits


# =========================================================
# CODE CHURN RATE
# =========================================================

def calculate_code_churn_rate(
    source_churn,
    target_source_loc
):
    """
    Operational definition:

    (Source lines added + deleted)
    / Total target source lines * 100

    Returns percentage.
    """

    if target_source_loc == 0:
        return 0.0

    return round(
        (source_churn / target_source_loc) * 100,
        6
    )


# =========================================================
# FEATURE GENERATION
# =========================================================

def generate_features(base_commit, target_commit="HEAD"):

    print("Running Git feature extractor...")

    git_output = run_command([
        sys.executable,
        str(GIT_EXTRACTOR),
        "--base",
        base_commit,
        "--target",
        target_commit
    ])

    git_data = json.loads(git_output)
    git_features = git_data["extracted_features"]

    print("Running static feature extractor...")

    static_output = run_command([
        sys.executable,
        str(STATIC_EXTRACTOR)
    ])

    static_data = json.loads(static_output)

    # -----------------------------------------------------
    # Changed files
    # -----------------------------------------------------

    changed_files = [
        item["file"]
        for item in git_data["changed_files"]
    ]

    # -----------------------------------------------------
    # Commit history
    # -----------------------------------------------------

    commit_count = count_commits_on_touched_files(
        base_commit,
        target_commit,
        changed_files
    )

    # -----------------------------------------------------
    # Target source LOC
    # -----------------------------------------------------

    target_source_loc = get_target_source_loc(
        target_commit
    )

    # -----------------------------------------------------
    # Code churn rate
    # -----------------------------------------------------

    code_churn_rate = calculate_code_churn_rate(
        git_features["git_diff_src_churn"],
        target_source_loc
    )

    # -----------------------------------------------------
    # Combine measured features
    # -----------------------------------------------------

    features = {

        "git_diff_src_churn":
            git_features["git_diff_src_churn"],

        "gh_diff_files_added":
            git_features["gh_diff_files_added"],

        "gh_diff_files_deleted":
            git_features["gh_diff_files_deleted"],

        "gh_diff_files_modified":
            git_features["gh_diff_files_modified"],

        "gh_diff_src_files":
            git_features["gh_diff_src_files"],

        "gh_diff_doc_files":
            git_features["gh_diff_doc_files"],

        "gh_diff_other_files":
            git_features["gh_diff_other_files"],

        "gh_sloc":
            static_data["gh_sloc"],

        "gh_num_commits_on_files_touched":
            commit_count,

        "code_churn_rate":
            code_churn_rate,

        "files_changed_total":
            git_features["files_changed_total"],

        "gh_diff_tests_added":
            git_features["gh_diff_tests_added"],

        "gh_diff_tests_deleted":
            git_features["gh_diff_tests_deleted"],

        "gh_test_lines_per_kloc":
            static_data["gh_test_lines_per_kloc"],

        "gh_test_cases_per_kloc":
            static_data["gh_test_cases_per_kloc"],

        "gh_asserts_cases_per_kloc":
            static_data["gh_asserts_cases_per_kloc"]

    }

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    if len(features) != 16:
        raise ValueError(
            f"Expected 16 measured features, got {len(features)}"
        )

    if target_source_loc == 0 and git_features["git_diff_src_churn"] > 0:
        raise ValueError(
            "Source churn exists but target source LOC is zero."
        )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\nTarget source LOC:", target_source_loc)

    print("\nGenerated features:")
    print(json.dumps(features, indent=4))

    print("\nFeature count:", len(features))

    print("\nNote:")
    print("16 measured features are currently available.")
    print("Three history features are still pending validation.")

    return features


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Live CI/CD pre-build feature generator"
    )

    parser.add_argument("--base", required=True)
    parser.add_argument("--target", default="HEAD")

    args = parser.parse_args()

    try:

        generated_features = generate_features(
            args.base,
            args.target
        )

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                generated_features,
                file,
                indent=4
            )

        print("\nFeature JSON saved successfully:")
        print(OUTPUT_FILE)

    except Exception as error:

        print(
            f"\nFeature generation failed: {error}",
            file=sys.stderr
        )

        sys.exit(1)
