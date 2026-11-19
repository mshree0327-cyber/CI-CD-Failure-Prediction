"""
Static feature extraction from a specific Git commit snapshot.

Operational scope:
- Python files only.
- Source/test classification uses git_feature_extractor.classify_file.
- gh_sloc counts nonblank, non-comment Python lines in source files.
- Test density metrics use nonblank, non-comment Python test lines,
  test_* function definitions, and `assert` statements in test files.
- Reads file contents with `git show COMMIT:path`; it does not inspect the
  current working tree. This keeps static metrics tied to target_commit.

These are transparent operational definitions. They are not proven identical
to the unavailable original TravisTorrent feature-generation code.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import PurePosixPath

from git_feature_extractor import classify_file

EXCLUDED_DIRECTORIES = {
    ".git", "venv", ".venv", "__pycache__", "node_modules",
    ".pytest_cache", ".mypy_cache", ".tox",
}


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Git command failed: git {' '.join(args)}\n{result.stderr.strip()}"
        )
    return result.stdout


def _is_excluded(path: PurePosixPath) -> bool:
    return any(part.lower() in EXCLUDED_DIRECTORIES for part in path.parts)


def _read_commit_file(commit: str, filename: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{filename}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _metrics_from_source(source: str) -> dict:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"lines": 0, "test_functions": 0, "assertions": 0}

    lines = [
        line for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    test_functions = sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in ast.walk(tree)
    )
    assertions = sum(
        isinstance(node, ast.Assert)
        for node in ast.walk(tree)
    )
    return {
        "lines": len(lines),
        "test_functions": test_functions,
        "assertions": assertions,
    }


def extract_static_features(target_commit: str = "HEAD") -> dict:
    # Validate target commit.
    git("rev-parse", "--verify", f"{target_commit}^{{commit}}")

    tracked_files = git(
        "ls-tree", "-r", "--name-only", target_commit
    ).splitlines()

    source_loc = 0
    test_lines = 0
    test_cases = 0
    assertions = 0
    source_files = 0
    test_files = 0
    skipped_non_python = 0
    skipped_unreadable = 0

    for filename in tracked_files:
        path = PurePosixPath(filename)
        if _is_excluded(path):
            continue
        if path.suffix.lower() != ".py":
            skipped_non_python += 1
            continue

        category = classify_file(filename)
        source = _read_commit_file(target_commit, filename)
        if source is None:
            skipped_unreadable += 1
            continue

        metrics = _metrics_from_source(source)
        if category == "test":
            test_files += 1
            test_lines += metrics["lines"]
            test_cases += metrics["test_functions"]
            assertions += metrics["assertions"]
        elif category == "source":
            source_files += 1
            source_loc += metrics["lines"]

    if source_loc:
        test_lines_per_kloc = test_lines / source_loc * 1000.0
        test_cases_per_kloc = test_cases / source_loc * 1000.0
        assertions_per_kloc = assertions / source_loc * 1000.0
    else:
        test_lines_per_kloc = 0.0
        test_cases_per_kloc = 0.0
        assertions_per_kloc = 0.0

    return {
        "target_commit": target_commit,
        "gh_sloc": source_loc,
        "gh_test_lines_per_kloc": test_lines_per_kloc,
        "gh_test_cases_per_kloc": test_cases_per_kloc,
        "gh_asserts_cases_per_kloc": assertions_per_kloc,
        "python_source_files": source_files,
        "python_test_files": test_files,
        "python_test_cases": test_cases,
        "python_assertions": assertions,
        "measurement_scope": "Tracked Python files at target commit; generated/cache directories excluded",
        "definitions": {
            "gh_sloc": "Nonblank, non-comment Python lines in files classified as source.",
            "gh_test_lines_per_kloc": "Nonblank, non-comment Python test lines / Python source LOC * 1000.",
            "gh_test_cases_per_kloc": "Function and async-function definitions named test_* in classified test files / Python source LOC * 1000.",
            "gh_asserts_cases_per_kloc": "Python assert statements in classified test files / Python source LOC * 1000.",
        },
        "limitations": [
            "Python-only metrics; other programming languages are not counted.",
            "Test classification depends on naming conventions and test/tests directory names.",
            "The original TravisTorrent feature-generation code is unavailable, so historical definition parity is unverified.",
        ],
        "diagnostics": {
            "tracked_python_source_files": source_files,
            "tracked_python_test_files": test_files,
            "skipped_non_python_files": skipped_non_python,
            "skipped_unreadable_files": skipped_unreadable,
        },
    }
