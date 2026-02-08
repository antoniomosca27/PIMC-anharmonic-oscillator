"""Plotting utilities for correlators and gap-analysis diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


def _as_1d(values: ArrayLike, *, name: str) -> np.ndarray:
    """Validate one-dimensional plotting input arrays."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def plot_correlator(
    tau: ArrayLike,
    G: ArrayLike,
    G_err: ArrayLike,
    *,
    G_exact: ArrayLike | None = None,
    title: str = "Euclidean Correlator",
) -> Figure:
    """Plot correlator data with error bars and optional analytic reference."""
    tau_arr = _as_1d(tau, name="tau")
    g_arr = _as_1d(G, name="G")
    g_err_arr = _as_1d(G_err, name="G_err")

    if not (tau_arr.size == g_arr.size == g_err_arr.size):
        raise ValueError("tau, G, and G_err must have the same length")

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.errorbar(
        tau_arr,
        g_arr,
        yerr=g_err_arr,
        fmt="o",
        markersize=4,
        linewidth=1.2,
        capsize=2,
        label="PIMC",
    )

    if G_exact is not None:
        g_exact_arr = _as_1d(G_exact, name="G_exact")
        if g_exact_arr.size != tau_arr.size:
            raise ValueError("G_exact must have the same length as tau")
        ax.plot(tau_arr, g_exact_arr, linewidth=1.8, label="Harmonic exact")

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$G(\tau)$")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_effective_mass(
    tau_mid: ArrayLike,
    Delta_eff: ArrayLike,
    *,
    title: str = "Effective Mass",
    fit_window: tuple[float, float] | None = None,
    Delta_eff_q16: ArrayLike | None = None,
    Delta_eff_q84: ArrayLike | None = None,
    band_label: str = "bootstrap 16-84%",
) -> Figure:
    """Plot effective mass (effective gap) as a function of Euclidean time.

    NaN entries in `Delta_eff` are plotted as gaps, which is useful when
    invalid late-time points are intentionally masked in the analysis stage.
    """
    tau_arr = _as_1d(tau_mid, name="tau_mid")
    delta_arr = _as_1d(Delta_eff, name="Delta_eff")

    if tau_arr.size != delta_arr.size:
        raise ValueError("tau_mid and Delta_eff must have the same length")

    q16_arr: np.ndarray | None = None
    q84_arr: np.ndarray | None = None
    if Delta_eff_q16 is not None or Delta_eff_q84 is not None:
        if Delta_eff_q16 is None or Delta_eff_q84 is None:
            raise ValueError(
                "Delta_eff_q16 and Delta_eff_q84 must be provided together"
            )
        q16_arr = _as_1d(Delta_eff_q16, name="Delta_eff_q16")
        q84_arr = _as_1d(Delta_eff_q84, name="Delta_eff_q84")
        if q16_arr.size != tau_arr.size or q84_arr.size != tau_arr.size:
            raise ValueError(
                "Delta_eff_q16 and Delta_eff_q84 must match tau_mid length"
            )

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    if q16_arr is not None and q84_arr is not None:
        ax.fill_between(
            tau_arr,
            q16_arr,
            q84_arr,
            color="tab:blue",
            alpha=0.2,
            linewidth=0.0,
            label=band_label,
        )
    ax.plot(
        tau_arr,
        delta_arr,
        marker="o",
        markersize=4,
        linewidth=1.4,
        label=r"$\Delta_{\mathrm{eff}}$",
    )
    if fit_window is not None:
        tau_min, tau_max = fit_window
        if tau_max <= tau_min:
            raise ValueError("fit_window must satisfy tau_max > tau_min")
        ax.axvspan(
            tau_min,
            tau_max,
            color="tab:gray",
            alpha=0.15,
            label="fit window",
        )
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\Delta_{\mathrm{eff}}(\tau)$")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_gap_vs_lambda(
    lams: ArrayLike,
    gaps: ArrayLike,
    gap_errs: ArrayLike,
    *,
    title: str = "Gap vs Quartic Coupling",
) -> Figure:
    """Plot extracted energy gaps as a function of quartic coupling."""
    lam_arr = _as_1d(lams, name="lams")
    gap_arr = _as_1d(gaps, name="gaps")
    gap_err_arr = _as_1d(gap_errs, name="gap_errs")

    if not (lam_arr.size == gap_arr.size == gap_err_arr.size):
        raise ValueError("lams, gaps, and gap_errs must have the same length")

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.errorbar(
        lam_arr,
        gap_arr,
        yerr=gap_err_arr,
        fmt="o-",
        capsize=3,
        linewidth=1.4,
        markersize=4,
    )
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\Delta$")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_histogram_with_gaussian(
    x_values: ArrayLike,
    mean: float,
    var: float,
    *,
    title: str = "Position Histogram",
) -> Figure:
    """Plot a normalized histogram and Gaussian reference density."""
    values = _as_1d(x_values, name="x_values")
    if values.size < 1:
        raise ValueError("x_values must contain at least one value")

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.hist(values, bins=50, density=True, alpha=0.75, label="Samples")

    if var > 0.0:
        sigma = float(np.sqrt(var))
        x_grid = np.linspace(np.min(values), np.max(values), 300)
        gaussian = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(
            -0.5 * ((x_grid - mean) / sigma) ** 2
        )
        ax.plot(x_grid, gaussian, linewidth=1.8, label="Gaussian")

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_path(
    path: ArrayLike,
    delta_tau: float | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Convenience helper to plot one Euclidean path."""
    path_arr = _as_1d(path, name="path")

    if ax is None:
        fig, axis = plt.subplots(figsize=(8, 4))
    else:
        fig, axis = ax.figure, ax

    tau = np.arange(path_arr.size, dtype=np.float64)
    if delta_tau is not None:
        tau = tau * delta_tau

    axis.plot(tau, path_arr, linewidth=1.4)
    axis.set_xlabel("Euclidean time")
    axis.set_ylabel("x")
    axis.set_title("Sampled path")
    return fig, axis


def plot_position_histogram(
    paths: ArrayLike,
    bins: int = 50,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Convenience helper to plot flattened position histograms."""
    values = np.asarray(paths, dtype=np.float64).ravel()

    if ax is None:
        fig, axis = plt.subplots(figsize=(8, 4))
    else:
        fig, axis = ax.figure, ax

    axis.hist(values, bins=bins, density=True, alpha=0.85)
    axis.set_xlabel("x")
    axis.set_ylabel("Density")
    axis.set_title("Position distribution")
    return fig, axis
