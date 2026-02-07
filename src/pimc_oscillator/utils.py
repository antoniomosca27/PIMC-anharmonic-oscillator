"""Shared utility helpers for seeding and small numeric operations."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from random import seed as random_seed

import numpy as np


def set_deterministic_seed(seed: int) -> int:
    """Configure deterministic seed metadata for the current process.

    The function seeds Python's `random` module and stores `PYTHONHASHSEED` in
    the process environment for reproducibility metadata (and for child
    processes spawned after this call). Python hash randomization for the
    current interpreter is fixed at process start and is not changed here.

    Args:
        seed: Non-negative integer seed.

    Returns:
        The validated seed value.
    """
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random_seed(seed)
    return seed


def make_rng(
    seed: int,
    *,
    set_hash_seed: bool = False,
) -> np.random.Generator:
    """Build a NumPy RNG backed by PCG64 with optional hash-seed setup."""
    if set_hash_seed:
        set_deterministic_seed(seed)
    bit_generator = np.random.PCG64(seed)
    return np.random.Generator(bit_generator)


def utc_timestamp() -> str:
    """Return a UTC ISO-8601 timestamp for metadata and logging."""
    return datetime.now(timezone.utc).isoformat()


def periodic_index(index: int, size: int) -> int:
    """Map an integer index to `[0, size)` using periodic wrapping."""
    if size <= 0:
        raise ValueError("size must be strictly positive")
    return index % size
