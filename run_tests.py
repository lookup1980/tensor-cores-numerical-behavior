#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
from pathlib import Path


TEST_RE = re.compile(r"^test-(\S+)$")


def discover_tests(root: Path):
    tests = []
    for path in sorted(root.iterdir()):
        match = TEST_RE.match(path.name)
        if match and path.is_file() and os.access(path, os.X_OK):
            tests.append((path, match.group(1)))
    return tests


def matches_selector(test_name: str, selector: str) -> bool:
    selector = selector.removeprefix("test-")
    return test_name == selector or test_name.startswith(selector + "-")


def select_tests(tests, selectors):
    if not selectors:
        return tests

    selected = []
    for test in tests:
        _, test_name = test
        if any(matches_selector(test_name, selector) for selector in selectors):
            selected.append(test)
    return selected


def result_path(output_dir: Path, test_name: str) -> Path:
    return output_dir / f"result-{test_name}.txt"


def require_repo_output_dir(output_dir: Path) -> None:
    repo = Path.cwd().resolve()
    resolved = output_dir.resolve()
    try:
        resolved.relative_to(repo)
    except ValueError:
        raise SystemExit("Output directory must be inside this repository")
def main():
    parser = argparse.ArgumentParser(
        description="Run tensor-core test binaries and write result files.")
    parser.add_argument(
        "selectors",
        nargs="*",
        help="GPU or test selector, e.g. 5090, test-5090, or 5090-bf16.")
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Directory for result-*.txt files.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected tests without running them.")
    args = parser.parse_args()

    root = Path(".")
    output_dir = Path(args.output_dir)
    require_repo_output_dir(output_dir)
    tests = select_tests(discover_tests(root), args.selectors)

    if not tests:
        selectors = ", ".join(args.selectors) if args.selectors else "all tests"
        raise SystemExit(f"No matching test binaries for: {selectors}")

    if args.dry_run:
        for _, test_name in tests:
            print(result_path(output_dir, test_name))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for path, test_name in tests:
        output = result_path(output_dir, test_name)
        print(output)
        with output.open("w") as outfile:
            subprocess.run([f"./{path.name}"], stdout=outfile, check=True)


if __name__ == "__main__":
    main()
