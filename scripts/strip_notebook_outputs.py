#!/usr/bin/env python3
"""Strip outputs and execution counts from one or more Jupyter notebooks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _strip_notebook_json(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True

    if changed:
        path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    return changed


def _strip_notebook_nbformat(path: Path) -> bool:
    import nbformat

    notebook = nbformat.read(path, as_version=nbformat.NO_CONVERT)
    changed = False

    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True

    if changed:
        nbformat.write(notebook, path)
    return changed


def strip_notebook(path: Path) -> bool:
    """Strip one notebook in-place and return whether it changed."""
    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")

    try:
        return _strip_notebook_nbformat(path)
    except ModuleNotFoundError:
        return _strip_notebook_json(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strip code-cell outputs and execution counts from notebooks.",
    )
    parser.add_argument(
        "notebooks",
        nargs="+",
        help="Notebook paths to clean in place.",
    )
    args = parser.parse_args(argv)

    changed_count = 0
    for raw_path in args.notebooks:
        path = Path(raw_path)
        changed = strip_notebook(path)
        changed_count += int(changed)
        status = "updated" if changed else "already clean"
        print(f"[strip-notebook] {path}: {status}")

    print(f"[strip-notebook] Processed {len(args.notebooks)} notebook(s).")
    print(f"[strip-notebook] Updated {changed_count} notebook(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
