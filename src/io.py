"""I/O helpers for experiment artifacts and run metadata."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.config import RunConfig

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Filesystem paths associated with one experiment run."""

    run_id: str
    logs_dir: Path
    reports_dir: Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory path if needed and return it as `Path`."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _ensure_parent(path: Path) -> None:
    """Create parent directories for a file path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _sanitize_run_name(run_name: str) -> str:
    """Sanitize run names for safe filesystem usage."""
    cleaned = run_name.strip().lower()
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", cleaned)
    cleaned = cleaned.strip("-_")
    return cleaned or "run"


def make_run_id(run_name: str | None = None) -> str:
    """Create run id with UTC millisecond timestamp and optional suffix."""
    now_utc = datetime.now(timezone.utc)
    millis = now_utc.microsecond // 1000
    timestamp = now_utc.strftime("%Y%m%dT%H%M%S") + f"{millis:03d}Z"
    if run_name is None:
        return timestamp
    return f"{timestamp}_{_sanitize_run_name(run_name)}"


def create_run_directories(
    *,
    out_root: str | Path = ".",
    run_name: str | None = None,
) -> RunPaths:
    """Create and return run-specific log/report directories."""
    run_id = make_run_id(run_name)
    root = Path(out_root)

    logs_dir = ensure_dir(root / "logs" / run_id)
    reports_dir = ensure_dir(root / "reports" / run_id)
    return RunPaths(run_id=run_id, logs_dir=logs_dir, reports_dir=reports_dir)


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Serialize a mapping to a JSON file with stable formatting."""
    target = Path(path)
    _ensure_parent(target)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    return json.loads(Path(path).read_text())


def save_array(array: ArrayLike, path: str | Path) -> Path:
    """Persist an array to `.npy` format."""
    target = Path(path)
    _ensure_parent(target)
    np.save(target, np.asarray(array, dtype=np.float64))
    return target


def load_array(path: str | Path) -> FloatArray:
    """Load an array from `.npy` format."""
    return np.load(Path(path))


def run_config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Serialize `RunConfig` into a JSON-compatible dictionary."""
    payload = asdict(config)
    payload["a"] = config.a
    payload["n_samples"] = config.n_samples
    payload["total_sweeps"] = config.total_sweeps
    return payload


def save_config_json(config: RunConfig, path: str | Path) -> Path:
    """Save run configuration to JSON."""
    return save_json(run_config_to_dict(config), path)


def save_chain_npz(
    *,
    samples: ArrayLike,
    metadata: dict[str, Any],
    path: str | Path,
) -> Path:
    """Persist chain samples and metadata to compressed NPZ."""
    target = Path(path)
    _ensure_parent(target)

    metadata_json = json.dumps(metadata, sort_keys=True)
    np.savez_compressed(
        target,
        samples=np.asarray(samples, dtype=np.float64),
        metadata=np.array(metadata_json),
    )
    return target


def save_analysis_json(analysis: dict[str, Any], path: str | Path) -> Path:
    """Save analysis summary payload to JSON."""
    return save_json(analysis, path)


def save_run_summary_json(summary: dict[str, Any], path: str | Path) -> Path:
    """Save compact human-readable run summary."""
    return save_json(summary, path)
