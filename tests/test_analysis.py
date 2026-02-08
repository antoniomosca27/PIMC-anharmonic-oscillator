"""Tests for effective-mass and gap-fit analysis routines."""

from __future__ import annotations

import numpy as np
from src.analysis import (
    effective_mass_bootstrap_from_correlators,
    effective_mass_from_correlator,
    fit_gap_cosh,
    harmonic_correlator,
    harmonic_x2,
    symmetrize_periodic_correlator,
)


def _synthetic_cosh_correlator(
    tau: np.ndarray,
    *,
    amplitude: float,
    delta: float,
    beta: float,
) -> np.ndarray:
    """Generate exact finite-temperature cosh correlator samples."""
    return amplitude * (
        np.exp(-delta * tau) + np.exp(-delta * (beta - tau))
    )


def test_effective_mass_recovers_known_gap_on_exact_cosh_data() -> None:
    """Effective mass should recover Delta for synthetic exact correlators."""
    beta = 6.0
    n_slices = 48
    a = beta / n_slices
    tau = np.arange(n_slices, dtype=np.float64) * a

    delta_true = 0.73
    g = _synthetic_cosh_correlator(tau, amplitude=1.1, delta=delta_true, beta=beta)

    _, delta_eff = effective_mass_from_correlator(g, a=a)

    # Exclude boundary-adjacent points where numerical cancellation is stronger.
    plateau = delta_eff[4:-4]
    assert np.allclose(plateau, delta_true, rtol=0.0, atol=1e-10)


def test_cosh_fit_recovers_gap_on_small_deterministic_noise() -> None:
    """Weighted cosh fit should recover the generating gap within tolerance."""
    beta = 6.0
    n_slices = 48
    a = beta / n_slices
    tau = np.arange(n_slices, dtype=np.float64) * a

    amplitude_true = 0.85
    delta_true = 1.05
    g_true = _synthetic_cosh_correlator(
        tau,
        amplitude=amplitude_true,
        delta=delta_true,
        beta=beta,
    )

    rng = np.random.Generator(np.random.PCG64(777))
    sigma = 2.5e-4
    g_noisy = g_true + sigma * rng.normal(size=n_slices)
    g_std = np.full(n_slices, sigma, dtype=np.float64)

    fit = fit_gap_cosh(
        tau=tau,
        G=g_noisy,
        G_std=g_std,
        beta=beta,
        fit_window=(10, 24),
    )

    assert fit.delta > 0.0
    assert abs(fit.delta - delta_true) < 0.05


def test_harmonic_x2_matches_correlator_at_tau_zero() -> None:
    """For the harmonic case, G(0) equals the analytic <x^2>."""
    beta = 3.4
    m = 1.2
    omega = 0.8

    g0 = harmonic_correlator(np.array([0.0]), beta=beta, m=m, omega=omega)[0]
    x2 = harmonic_x2(beta=beta, m=m, omega=omega)

    assert np.isclose(g0, x2, rtol=0.0, atol=1e-12)


def test_effective_mass_invalid_points_can_be_marked_nan() -> None:
    """Invalid arcosh ratios should become NaN when requested."""
    g = np.array([1.0, 3.0, 1.0, 3.0, 1.0], dtype=np.float64)
    _, delta_default = effective_mass_from_correlator(g, a=0.2)
    _, delta_nan = effective_mass_from_correlator(g, a=0.2, invalid_as_nan=True)

    assert np.all(np.isfinite(delta_default))
    assert np.isnan(delta_nan[0])
    assert np.isnan(delta_nan[2])


def test_effective_mass_supports_snr_masking() -> None:
    """Low-SNR points should be masked when G_err and snr_min are supplied."""
    beta = 5.0
    n_slices = 40
    a = beta / n_slices
    tau = np.arange(n_slices, dtype=np.float64) * a
    g = _synthetic_cosh_correlator(tau, amplitude=1.2, delta=0.8, beta=beta)

    g_err = np.full_like(g, 1e-3)
    g_err[-6:] = 10.0  # Force low-SNR late-time points.

    _, delta_eff = effective_mass_from_correlator(
        g,
        a=a,
        invalid_as_nan=True,
        G_err=g_err,
        snr_min=5.0,
    )

    assert np.isfinite(delta_eff[5])
    assert np.isnan(delta_eff[-1])


def test_symmetrize_periodic_correlator_enforces_time_reflection_symmetry() -> None:
    """Symmetrized correlators satisfy G[k] == G[N-k] for all k."""
    g = np.array([3.0, 2.0, 7.0, 5.0, 4.0, 9.0, 1.0, 6.0], dtype=np.float64)
    g_sym = symmetrize_periodic_correlator(g)
    n = g_sym.size

    for k in range(n):
        assert np.isclose(g_sym[k], g_sym[(n - k) % n])


def test_effective_mass_bootstrap_summary_recovers_plateau() -> None:
    """Bootstrap summaries should reproduce a clean synthetic plateau."""
    beta = 6.0
    n_slices = 48
    a = beta / n_slices
    tau = np.arange(n_slices, dtype=np.float64) * a

    delta_true = 0.91
    g = _synthetic_cosh_correlator(tau, amplitude=1.1, delta=delta_true, beta=beta)
    boot = np.repeat(g[np.newaxis, :], repeats=20, axis=0)

    summary = effective_mass_bootstrap_from_correlators(boot, a=a)

    plateau = summary.mean[4:-4]
    assert np.allclose(plateau, delta_true, rtol=0.0, atol=1e-10)
    assert np.allclose(summary.std[4:-4], 0.0, rtol=0.0, atol=1e-12)
    assert np.allclose(summary.q16[4:-4], delta_true, rtol=0.0, atol=1e-10)
    assert np.allclose(summary.q84[4:-4], delta_true, rtol=0.0, atol=1e-10)


def test_effective_mass_bootstrap_all_nan_point_returns_nan_summary() -> None:
    """A tau point with no valid replicas should remain NaN in all summaries."""
    beta = 4.0
    n_slices = 24
    a = beta / n_slices
    tau = np.arange(n_slices, dtype=np.float64) * a

    g = _synthetic_cosh_correlator(tau, amplitude=0.9, delta=0.8, beta=beta)
    boot = np.repeat(g[np.newaxis, :], repeats=6, axis=0)
    boot[:, 7] = -1.0  # Force invalid arcosh ratio at one interior tau point.

    summary = effective_mass_bootstrap_from_correlators(
        boot,
        a=a,
        invalid_as_nan=True,
    )

    interior_index = 6
    assert np.isnan(summary.mean[interior_index])
    assert np.isnan(summary.std[interior_index])
    assert np.isnan(summary.q16[interior_index])
    assert np.isnan(summary.q84[interior_index])
    assert summary.n_valid[interior_index] == 0
