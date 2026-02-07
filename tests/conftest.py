"""Shared pytest fixtures and test configuration."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from pimc_oscillator import LatticeParams, PotentialParams

# Ensure tests never require an interactive display backend.
matplotlib.use("Agg", force=True)


@pytest.fixture
def rng() -> np.random.Generator:
    """Provide a deterministic RNG for repeatable numeric tests."""
    return np.random.Generator(np.random.PCG64(20240207))


@pytest.fixture
def lattice_params() -> LatticeParams:
    """Representative lattice parameters used across fast unit tests."""
    return LatticeParams(beta=2.4, n_slices=16)


@pytest.fixture
def potential_params() -> PotentialParams:
    """Representative potential parameters used across fast unit tests."""
    return PotentialParams(m=1.3, omega=0.9, lam=0.35)
