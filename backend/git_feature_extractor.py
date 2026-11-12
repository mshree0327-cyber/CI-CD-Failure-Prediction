
import argparse
import json
import subprocess
from pathlib import Path


SOURCE_EXTENSIONS = {
    ".py", ".java", ".js", ".ts", ".jsx",
    ".tsx", ".c", ".cpp", ".h", ".hpp",
    ".php", ".cs", ".go", ".rb"
}

TEST_EXTENSIONS = SOURCE_EXTENSIONS

DOCUMENTATION_EXTENSIONS = {
    ".md", ".txt", ".rst"
}


def git(*args):
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8"
    )
    return result.stdout


def classify_file(filename):
    path = Path(filename)
    name = path.name.lower()
    suffix = path.suffix.lower()
    parts = {part.lower() for part in path.parts}

    is_test = (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".test.js")
        or name.endswith(".test.ts")
        or name.endswith(".spec.js")
        or name.endswith(".spec.ts")
        or "tests" in parts
        or "test" in parts
    )

    if is_test and suffix in TEST_EXTENSIONS:
        return "test"

    if suffix in DOCUMENTATION_EXTENSIONS:
        return "doc"

    if suffix in SOURCE_EXTENSIONS:
        return "source"

    return "other"


def get_line_churn(base, target):
    output = git(
        "diff", "--numstat",
        "--no-renames", base, target
    )

    source_added = 0
    source_deleted = 0
    test_added = 0
    test_deleted = 0

    for line in output.splitlines():
        if not line.strip():
            continue

        added, deleted, filename = line.split("\t", 2)

        # Binary files are represented by '-'
        if added == "-" or deleted == "-":
            continue

        added = int(added)
        deleted = int(deleted)

        category = classify_file(filename)

        if category == "source":
            source_added += added
            source_deleted += deleted

        elif category == "test":
            test_added += added
            test_deleted += deleted

    return {
        "source_lines_added": source_added,
        "source_lines_deleted": source_deleted,
        "git_diff_src_churn": source_added + source_deleted,
        "test_lines_added": test_added,
        "test_lines_deleted": test_deleted
    }


def extract_features(base, target):
    diff = git(
        "diff", "--name-status",
        "--no-renames", base, target
    )

    features = {
        "gh_diff_files_added": 0,
        "gh_diff_files_deleted": 0,
        "gh_diff_files_modified": 0,
        "gh_diff_src_files": 0,
        "gh_diff_doc_files": 0,
        "gh_diff_other_files": 0,
        "gh_diff_tests_added": 0,
        "gh_diff_tests_deleted": 0,
        "files_changed_total": 0
    }

    changed_files = []

    for line in diff.splitlines():
        if not line.strip():
            continue

        status, filename = line.split("\t", 1)
        category = classify_file(filename)

        features["files_changed_total"] += 1

        if status == "A":
            features["gh_diff_files_added"] += 1
        elif status == "D":
            features["gh_diff_files_deleted"] += 1
        elif status == "M":
            features["gh_diff_files_modified"] += 1

        if category == "source":
            features["gh_diff_src_files"] += 1

        elif category == "test":
            if status == "A":
                features["gh_diff_tests_added"] += 1
            elif status == "D":
                features["gh_diff_tests_deleted"] += 1

        elif category == "doc":
            features["gh_diff_doc_files"] += 1

        else:
            features["gh_diff_other_files"] += 1

        changed_files.append({
            "file": filename,
            "status": status,
            "category": category
        })

    churn = get_line_churn(base, target)

    features.update(churn)

    return {
        "base_commit": base,
        "target_commit": target,
        "extracted_features": features,
        "changed_files": changed_files,
        "definitions": {
            "git_diff_src_churn": (
                "Source lines added plus source lines deleted. "
                "Operational definition; not verified as the "
                "original TravisTorrent formula."
            ),
            "gh_diff_tests_added": (
                "Number of added files classified as test files."
            ),
            "gh_diff_tests_deleted": (
                "Number of deleted files classified as test files."
            )
        },
        "note": (
            "Git-derived features only. This is not yet a "
            "complete 19-feature model payload."
        )
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--target", default="HEAD")
    args = parser.parse_args()

    result = extract_features(args.base, args.target)

    print(json.dumps(result, indent=4))
