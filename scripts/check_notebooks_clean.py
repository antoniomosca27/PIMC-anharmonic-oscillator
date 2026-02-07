#!/usr/bin/env python3
"""Fail if notebooks contain code-cell outputs or execution counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check_notebook(path: Path) -> list[str]:
    """Return validation errors for one notebook."""
    if not path.exists():
        return [f"{path}: file not found"]

    notebook = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []

    for idx, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        execution_count = cell.get("execution_count")
        outputs = cell.get("outputs", [])

        if execution_count is not None:
            errors.append(
                f"{path}: cell {idx} has execution_count={execution_count!r} "
                "(expected null)"
            )
        if outputs:
            errors.append(
                f"{path}: cell {idx} has {len(outputs)} output(s) (expected empty list)"
            )

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check that notebooks are committed without outputs.",
    )
    parser.add_argument("notebooks", nargs="+", help="Notebook paths to validate.")
    args = parser.parse_args(argv)

    all_errors: list[str] = []
    for raw_path in args.notebooks:
        all_errors.extend(check_notebook(Path(raw_path)))

    if all_errors:
        print("[check-notebooks] Notebook cleanliness check failed:")
        for error in all_errors:
            print(f"  - {error}")
        return 1

    print("[check-notebooks] All notebooks are clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
