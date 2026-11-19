"""
Git-based pre-build feature extraction.

Operational definitions:
- File counts and line churn are calculated from BASE..TARGET.
- Source/test/doc classification follows the rules below.
- gh_num_commits_on_files_touched counts UNIQUE commits in BASE..TARGET
  that touched at least one changed file (not a sum of per-file counts).
- code_churn_rate = source lines added + deleted / elapsed days between
  BASE and TARGET commit timestamps.

These are documented operational definitions, not a claim that they
reproduce the unavailable original TravisTorrent feature-generation code.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

SOURCE_EXTENSIONS = {
    ".py", ".java", ".js", ".ts", ".jsx", ".tsx", ".c", ".cpp", ".h",
    ".hpp", ".php", ".cs", ".go", ".rb",
}
DOCUMENTATION_EXTENSIONS = {".md", ".txt", ".rst"}
TEST_DIRECTORY_NAMES = {"test", "tests"}
TEST_FILE_SUFFIXES = (
    "_test.py", "test.java", ".test.js", ".test.ts", ".spec.js", ".spec.ts",
    "_test.go", "_spec.rb",
)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Git command failed: git {' '.join(args)}\n{result.stderr.strip()}"
        )
    return result.stdout


def classify_file(filename: str) -> str:
    path = Path(filename.replace("\\", "/"))
    name = path.name.lower()
    suffix = path.suffix.lower()
    parts = {p.lower() for p in path.parts}

    is_test = (
        name.startswith("test_")
        or any(name.endswith(s) for s in TEST_FILE_SUFFIXES)
        or bool(parts & TEST_DIRECTORY_NAMES)
    )
    if is_test and suffix in SOURCE_EXTENSIONS:
        return "test"
    if suffix in DOCUMENTATION_EXTENSIONS:
        return "doc"
    if suffix in SOURCE_EXTENSIONS:
        return "source"
    return "other"


def commit_timestamp(revision: str) -> int:
    return int(git("show", "-s", "--format=%ct", revision).strip())


def extract_features(base: str, target: str = "HEAD") -> dict:
    # Validate revisions and establish a non-negative elapsed time.
    base_ts = commit_timestamp(base)
    target_ts = commit_timestamp(target)
    elapsed_days = max((target_ts - base_ts) / 86400.0, 0.0)

    name_status = git("diff", "--name-status", "--no-renames", base, target)
    numstat = git("diff", "--numstat", "--no-renames", base, target)

    counts = {
        "gh_diff_files_added": 0,
        "gh_diff_files_deleted": 0,
        "gh_diff_files_modified": 0,
        "gh_diff_src_files": 0,
        "gh_diff_doc_files": 0,
        "gh_diff_other_files": 0,
        "gh_diff_tests_added": 0,
        "gh_diff_tests_deleted": 0,
        "files_changed_total": 0,
    }
    changed_files = []
    changed_paths = []

    for line in name_status.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        # --no-renames means ordinary records have status + one path.
        filename = parts[-1]
        category = classify_file(filename)
        status_code = status[0]
        counts["files_changed_total"] += 1
        changed_paths.append(filename)

        if status_code == "A":
            counts["gh_diff_files_added"] += 1
        elif status_code == "D":
            counts["gh_diff_files_deleted"] += 1
        elif status_code == "M":
            counts["gh_diff_files_modified"] += 1

        if category == "source":
            counts["gh_diff_src_files"] += 1
        elif category == "test":
            if status_code == "A":
                counts["gh_diff_tests_added"] += 1
            elif status_code == "D":
                counts["gh_diff_tests_deleted"] += 1
        elif category == "doc":
            counts["gh_diff_doc_files"] += 1
        else:
            counts["gh_diff_other_files"] += 1

        changed_files.append({
            "file": filename, "status": status, "category": category
        })

    source_added = source_deleted = 0
    test_added = test_deleted = 0
    for line in numstat.splitlines():
        if not line.strip():
            continue
        added, deleted, filename = line.split("\t", 2)
        if added == "-" or deleted == "-":  # binary file
            continue
        a, d = int(added), int(deleted)
        category = classify_file(filename)
        if category == "source":
            source_added += a
            source_deleted += d
        elif category == "test":
            test_added += a
            test_deleted += d

    source_churn = source_added + source_deleted
    test_churn_added = test_added
    test_churn_deleted = test_deleted

    unique_commits = set()
    if changed_paths:
        # One Git log query across all changed paths; deduplicate commit hashes.
        # Use -- to separate revisions from pathspecs.
        log = git(
            "log", "--format=%H", f"{base}..{target}", "--", *changed_paths
        )
        unique_commits = {x.strip() for x in log.splitlines() if x.strip()}

    code_churn_rate = source_churn / elapsed_days if elapsed_days > 0 else 0.0

    counts.update({
        "git_diff_src_churn": source_churn,
        "gh_num_commits_on_files_touched": len(unique_commits),
        "code_churn_rate": code_churn_rate,
        # Internal diagnostics, not model inputs:
        "_source_lines_added": source_added,
        "_source_lines_deleted": source_deleted,
        "_test_lines_added": test_churn_added,
        "_test_lines_deleted": test_churn_deleted,
        "_elapsed_days": elapsed_days,
    })

    return {
        "base_commit": base,
        "target_commit": target,
        "extracted_features": counts,
        "changed_files": changed_files,
        "definitions": {
            "git_diff_src_churn": "Source lines added plus source lines deleted in BASE..TARGET.",
            "code_churn_rate": "Source line churn divided by elapsed days between BASE and TARGET commit timestamps; lines/day.",
            "gh_num_commits_on_files_touched": "Number of unique commits in BASE..TARGET touching at least one changed path.",
            "gh_diff_tests_added": "Number of added files classified as test files.",
            "gh_diff_tests_deleted": "Number of deleted files classified as test files.",
        },
        "limitations": [
            "Feature definitions are operational approximations because the original feature-generation code is unavailable.",
            "File classification depends on extensions and test naming/directory conventions.",
            "A zero or negative BASE-to-TARGET time interval produces code_churn_rate=0.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--target", default="HEAD")
    parser.add_argument("--output", default="git_features.json")
    args = parser.parse_args()
    result = extract_features(args.base, args.target)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
