
import argparse
import json
import subprocess
from pathlib import Path


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

    source_extensions = {
        ".py", ".java", ".js", ".ts", ".jsx",
        ".tsx", ".c", ".cpp", ".h", ".hpp",
        ".php", ".cs", ".go", ".rb"
    }

    documentation_extensions = {
        ".md", ".txt", ".rst"
    }

    if name.startswith("test_") or name.endswith("_test.py"):
        return "test"

    if suffix in documentation_extensions:
        return "doc"

    if suffix in source_extensions:
        return "source"

    return "other"


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

        if category in ("source", "test"):
            features["gh_diff_src_files"] += 1
        elif category == "doc":
            features["gh_diff_doc_files"] += 1
        else:
            features["gh_diff_other_files"] += 1

        changed_files.append({
            "file": filename,
            "status": status,
            "category": category
        })

    return {
        "base_commit": base,
        "target_commit": target,
        "extracted_features": features,
        "changed_files": changed_files,
        "note": (
            "These are Git-derived measurements. "
            "They are not yet a complete 19-feature "
            "TravisTorrent-compatible payload."
        )
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--target", default="HEAD")
    args = parser.parse_args()

    result = extract_features(args.base, args.target)

    print(json.dumps(result, indent=4))
