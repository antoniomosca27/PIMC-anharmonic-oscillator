"""Tests for correlator estimators and bootstrap wrappers."""

from __future__ import annotations

import numpy as np

from src.observables import (
    bootstrap_correlator,
    correlator_single_configuration,
)


def _naive_translational_correlator(x: np.ndarray) -> np.ndarray:
    """Reference O(N^2) implementation for translational correlator g_k(x)."""
    n = x.size
    out = np.empty(n, dtype=np.float64)
    for k in range(n):
        acc = 0.0
        for i in range(n):
            acc += x[i] * x[(i + k) % n]
        out[k] = acc / float(n)
    return out


def test_fft_correlator_matches_naive_implementation() -> None:
    """FFT correlator must match direct summation for a small test vector."""
    n = 16
    x = np.linspace(-1.2, 1.1, n, dtype=np.float64)

    g_fft = correlator_single_configuration(x, subtract_mean=False)
    g_naive = _naive_translational_correlator(x)

    assert np.allclose(g_fft, g_naive, rtol=0.0, atol=1e-12)


def test_bootstrap_correlator_is_deterministic_with_fixed_seed(
    rng: np.random.Generator,
) -> None:
    """Bootstrap outputs should be reproducible for a fixed RNG seed."""
    paths = rng.normal(loc=0.0, scale=1.0, size=(24, 16))

    estimate_a = bootstrap_correlator(
        paths,
        subtract_mean=False,
        bin_size=4,
        n_boot=40,
        seed=202,
        return_quantiles=True,
        return_bootstrap_samples=False,
    )
    estimate_b = bootstrap_correlator(
        paths,
        subtract_mean=False,
        bin_size=4,
        n_boot=40,
        seed=202,
        return_quantiles=True,
        return_bootstrap_samples=False,
    )

    assert np.allclose(estimate_a.G_mean, estimate_b.G_mean)
    assert np.allclose(estimate_a.G_std, estimate_b.G_std)
    assert estimate_a.G_q16 is not None
    assert estimate_a.G_q84 is not None
    assert np.allclose(estimate_a.G_q16, estimate_b.G_q16)
    assert np.allclose(estimate_a.G_q84, estimate_b.G_q84)
