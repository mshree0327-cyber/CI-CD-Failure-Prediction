
import ast
import json
from pathlib import Path

from git_feature_extractor import classify_file


SOURCE_EXTENSIONS = {".py"}

EXCLUDED_DIRECTORIES = {
    ".git",
    "venv",
    "__pycache__",
    "node_modules",
    "back backup",
    "backend_backup",
    "data backup",
    "front backup",
    "extractor_test",
    "extractor_testmkdir",
}


def count_python_metrics(path):
    """Count Python lines and AST test metrics without executing code."""

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return {
            "lines": 0,
            "test_functions": 0,
            "assertions": 0,
        }

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


def extract_static_features(repository_root):
    root = Path(repository_root).resolve()

    source_loc = 0
    test_lines = 0
    test_cases = 0
    assertions = 0
    source_files = 0
    test_files = 0

    for path in root.rglob("*.py"):
        relative = path.relative_to(root)

        if any(
            part.lower() in EXCLUDED_DIRECTORIES
            for part in relative.parts
        ):
            continue

        category = classify_file(str(relative))
        metrics = count_python_metrics(path)

        if category == "test":
            test_files += 1
            test_lines += metrics["lines"]
            test_cases += metrics["test_functions"]
            assertions += metrics["assertions"]

        elif category == "source":
            source_files += 1
            source_loc += metrics["lines"]

    if source_loc:
        test_lines_per_kloc = test_lines / source_loc * 1000
        test_cases_per_kloc = test_cases / source_loc * 1000
        assertions_per_kloc = assertions / source_loc * 1000
    else:
        test_lines_per_kloc = 0
        test_cases_per_kloc = 0
        assertions_per_kloc = 0

    return {
        "gh_sloc": source_loc,
        "gh_test_lines_per_kloc": test_lines_per_kloc,
        "gh_test_cases_per_kloc": test_cases_per_kloc,
        "gh_asserts_cases_per_kloc": assertions_per_kloc,
        "python_source_files": source_files,
        "python_test_files": test_files,
        "python_test_cases": test_cases,
        "python_assertions": assertions,
        "measurement_scope": "Python files only",
        "excluded_directories": sorted(EXCLUDED_DIRECTORIES),
        "definitions": {
            "gh_sloc": "Nonblank, non-comment Python lines",
            "test_cases": "Function definitions starting with test_",
            "assertions": "Python assert statements in classified test files",
        },
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    result = extract_static_features(root)
    print(json.dumps(result, indent=4))
