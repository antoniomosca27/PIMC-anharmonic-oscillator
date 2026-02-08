"""Analysis tools for correlators, effective mass, and gap extraction."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class GapFitResult:
    """Result of a weighted cosh fit to a correlator window."""

    amplitude: float
    amplitude_std: float
    delta: float
    delta_std: float
    covariance: FloatArray
    fit_indices: IntArray


@dataclass(frozen=True, slots=True)
class GapBootstrapResult:
    """Bootstrap summary for gap estimates."""

    delta_mean: float
    delta_std: float
    delta_q16: float
    delta_q84: float
    deltas: FloatArray
    n_success: int


@dataclass(frozen=True, slots=True)
class EffectiveMassBootstrapResult:
    """Bootstrap summary for effective-mass curves."""

    tau_mid: FloatArray
    mean: FloatArray
    std: FloatArray
    q16: FloatArray
    q84: FloatArray
    n_valid: IntArray


def harmonic_correlator(
    tau: ArrayLike,
    *,
    beta: float,
    m: float,
    omega: float,
) -> FloatArray:
    """Analytic Euclidean correlator for the harmonic oscillator (lam=0).

    Formula:
        G_exact(tau) = [1 / (2*m*omega)]
                       * cosh(omega*(beta/2 - tau))
                       / sinh(beta*omega/2)
    """
    if beta <= 0.0:
        raise ValueError("beta must be strictly positive")
    if m <= 0.0:
        raise ValueError("m must be strictly positive")
    if omega <= 0.0:
        raise ValueError("omega must be strictly positive")

    tau_arr = np.asarray(tau, dtype=np.float64)
    z = 0.5 * beta * omega
    prefactor = 1.0 / (2.0 * m * omega)
    numerator = np.cosh(omega * (0.5 * beta - tau_arr))
    denominator = np.sinh(z)
    return prefactor * numerator / denominator


def harmonic_x2(*, beta: float, m: float, omega: float) -> float:
    """Analytic harmonic result for <x^2> (lam=0).

    Formula:
        <x^2> = [1 / (2*m*omega)] * coth(beta*omega/2)
    """
    if beta <= 0.0:
        raise ValueError("beta must be strictly positive")
    if m <= 0.0:
        raise ValueError("m must be strictly positive")
    if omega <= 0.0:
        raise ValueError("omega must be strictly positive")

    z = 0.5 * beta * omega
    coth = np.cosh(z) / np.sinh(z)
    return float((1.0 / (2.0 * m * omega)) * coth)


def symmetrize_periodic_correlator(G: ArrayLike) -> FloatArray:
    """Symmetrize a periodic correlator under tau <-> beta - tau.

    For a correlator of length N:
        G_sym[k] = 0.5 * (G[k] + G[N-k]),
    where indexing is periodic and `G[N] == G[0]`.
    """
    values = np.asarray(G, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("G must be one-dimensional")
    if values.size < 2:
        raise ValueError("G must contain at least 2 points")

    mirrored = values[(-np.arange(values.size)) % values.size]
    return 0.5 * (values + mirrored)


def effective_mass_from_correlator(
    G: ArrayLike,
    *,
    a: float,
    eps: float = 1e-12,
    invalid_as_nan: bool = False,
    G_err: ArrayLike | None = None,
    snr_min: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compute effective masses from the arcosh ratio estimator.

    For lattice point k = 1..N-2:
        ratio_k = (G_{k-1} + G_{k+1}) / (2*G_k)
        Delta_eff(k*a) = arcosh(max(ratio_k, 1+eps)) / a

    By default, invalid ratios are clipped to keep finite outputs. If
    `invalid_as_nan=True`, invalid points are returned as
    NaN instead of clipping. If `G_err` and `snr_min` are provided, points
    with signal-to-noise below the threshold are marked invalid.
    """
    if a <= 0.0:
        raise ValueError("a must be strictly positive")

    values = np.asarray(G, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("G must be one-dimensional")
    if values.size < 3:
        raise ValueError("G must contain at least 3 points")
    if snr_min is not None and snr_min <= 0.0:
        raise ValueError("snr_min must be strictly positive when provided")

    err_values: np.ndarray | None = None
    if G_err is not None:
        err_values = np.asarray(G_err, dtype=np.float64)
        if err_values.ndim != 1:
            raise ValueError("G_err must be one-dimensional")
        if err_values.size != values.size:
            raise ValueError("G_err must have the same length as G")

    denom = 2.0 * values[1:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (values[:-2] + values[2:]) / denom

    valid = np.isfinite(ratio)
    if snr_min is not None and err_values is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = values[1:-1] / err_values[1:-1]
        valid &= np.isfinite(snr) & (snr > snr_min)

    if invalid_as_nan:
        valid &= ratio >= (1.0 + eps)
        delta_eff = np.full_like(ratio, np.nan, dtype=np.float64)
        delta_eff[valid] = np.arccosh(ratio[valid]) / a
    else:
        ratio = np.where(valid, ratio, 1.0 + eps)
        ratio = np.maximum(ratio, 1.0 + eps)
        delta_eff = np.arccosh(ratio) / a

    tau_mid = np.arange(1, values.size - 1, dtype=np.float64) * a
    return tau_mid, delta_eff


def effective_mass_bootstrap_from_correlators(
    bootstrap_correlators: ArrayLike,
    *,
    a: float,
    eps: float = 1e-12,
    invalid_as_nan: bool = True,
    G_err: ArrayLike | None = None,
    snr_min: float | None = None,
) -> EffectiveMassBootstrapResult:
    """Estimate effective-mass summaries from bootstrap correlator replicates.

    The input `bootstrap_correlators` must have shape `(n_boot, N)`. Effective
    masses are computed along axis 1 and summarized across replicates while
    ignoring NaNs.
    """
    if a <= 0.0:
        raise ValueError("a must be strictly positive")

    boot = np.asarray(bootstrap_correlators, dtype=np.float64)
    if boot.ndim != 2:
        raise ValueError("bootstrap_correlators must be a 2D array")
    if boot.shape[0] < 1:
        raise ValueError("bootstrap_correlators must contain at least one replicate")
    if boot.shape[1] < 3:
        raise ValueError("bootstrap_correlators must have at least 3 time points")
    if snr_min is not None and snr_min <= 0.0:
        raise ValueError("snr_min must be strictly positive when provided")

    err_values: np.ndarray | None = None
    if G_err is not None:
        err_values = np.asarray(G_err, dtype=np.float64)
        if err_values.ndim == 1:
            if err_values.size != boot.shape[1]:
                raise ValueError(
                    "1D G_err must have the same length as correlator time points"
                )
        elif err_values.ndim == 2:
            if err_values.shape != boot.shape:
                raise ValueError(
                    "2D G_err must have the same shape as bootstrap_correlators"
                )
        else:
            raise ValueError("G_err must be one-dimensional or two-dimensional")

    denom = 2.0 * boot[:, 1:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (boot[:, :-2] + boot[:, 2:]) / denom

    valid = np.isfinite(ratio)
    if snr_min is not None and err_values is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            if err_values.ndim == 1:
                snr = boot[:, 1:-1] / err_values[np.newaxis, 1:-1]
            else:
                snr = boot[:, 1:-1] / err_values[:, 1:-1]
        valid &= np.isfinite(snr) & (snr > snr_min)

    if invalid_as_nan:
        valid &= ratio >= (1.0 + eps)
        delta_eff = np.full_like(ratio, np.nan, dtype=np.float64)
        delta_eff[valid] = np.arccosh(ratio[valid]) / a
    else:
        ratio = np.where(valid, ratio, 1.0 + eps)
        ratio = np.maximum(ratio, 1.0 + eps)
        delta_eff = np.arccosh(ratio) / a

    n_valid = np.sum(np.isfinite(delta_eff), axis=0, dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(delta_eff, axis=0)
        std = np.nanstd(delta_eff, axis=0, ddof=1)
        q16, q84 = np.nanpercentile(delta_eff, [16.0, 84.0], axis=0)

    mean = np.where(n_valid > 0, mean, np.nan).astype(np.float64)
    std = np.where(n_valid > 1, std, np.where(n_valid == 1, 0.0, np.nan)).astype(
        np.float64
    )
    q16 = np.where(n_valid > 0, q16, np.nan).astype(np.float64)
    q84 = np.where(n_valid > 0, q84, np.nan).astype(np.float64)

    tau_mid = np.arange(1, boot.shape[1] - 1, dtype=np.float64) * a
    return EffectiveMassBootstrapResult(
        tau_mid=tau_mid,
        mean=mean,
        std=std,
        q16=q16,
        q84=q84,
        n_valid=np.asarray(n_valid, dtype=np.int64),
    )


def default_fit_window_indices(
    *,
    beta: float,
    a: float,
    n_slices: int,
    tau_min_frac: float = 0.2,
    tau_max_frac: float = 0.5,
) -> tuple[int, int]:
    """Choose a default fit window as fractions of beta.

    Default window is `[0.2*beta, 0.5*beta]` mapped to lattice indices.
    """
    if beta <= 0.0:
        raise ValueError("beta must be strictly positive")
    if a <= 0.0:
        raise ValueError("a must be strictly positive")
    if n_slices < 4:
        raise ValueError("n_slices must be >= 4")
    if not 0.0 <= tau_min_frac < tau_max_frac <= 1.0:
        raise ValueError("tau_min_frac and tau_max_frac must satisfy 0<=min<max<=1")

    k_min = int(np.ceil((tau_min_frac * beta) / a))
    k_max = int(np.floor((tau_max_frac * beta) / a))

    k_min = max(1, k_min)
    k_max = min(n_slices - 2, k_max)

    if (k_max - k_min + 1) < 3:
        # Enforce at least three points for a stable two-parameter fit.
        k_min = max(1, min(k_min, n_slices - 4))
        k_max = min(n_slices - 2, k_min + 2)

    return k_min, k_max


def _cosh_model(
    tau: FloatArray,
    amplitude: float,
    delta: float,
    beta: float,
) -> FloatArray:
    """Two-sided finite-temperature exponential model."""
    return amplitude * (
        np.exp(-delta * tau) + np.exp(-delta * (beta - tau))
    )


def _resolve_fit_indices(
    tau: FloatArray,
    *,
    beta: float,
    fit_window: tuple[int, int] | tuple[float, float] | None,
) -> IntArray:
    """Resolve fit window specification to explicit index array."""
    n = tau.size
    if n < 4:
        raise ValueError("tau must have at least 4 points")

    a = float(tau[1] - tau[0])
    if a <= 0.0:
        raise ValueError("tau grid must be strictly increasing")

    if fit_window is None:
        k_min, k_max = default_fit_window_indices(beta=beta, a=a, n_slices=n)
        return np.arange(k_min, k_max + 1, dtype=np.int64)

    lo, hi = fit_window
    if isinstance(lo, (int, np.integer)) and isinstance(hi, (int, np.integer)):
        k_min = max(1, int(lo))
        k_max = min(n - 2, int(hi))
        if k_max <= k_min:
            raise ValueError("fit_window indices must satisfy hi > lo")
        return np.arange(k_min, k_max + 1, dtype=np.int64)

    tau_min = float(lo)
    tau_max = float(hi)
    if not isfinite(tau_min) or not isfinite(tau_max):
        raise ValueError("fit_window bounds must be finite")
    if tau_max <= tau_min:
        raise ValueError("fit_window bounds must satisfy tau_max > tau_min")

    mask = (tau >= tau_min) & (tau <= tau_max)
    mask &= np.arange(n) >= 1
    mask &= np.arange(n) <= n - 2

    indices = np.flatnonzero(mask).astype(np.int64)
    if indices.size < 3:
        raise ValueError("fit window must contain at least three interior points")
    return indices


def _safe_sigma(values: FloatArray) -> FloatArray:
    """Make fit uncertainties strictly positive and finite."""
    sigma = np.asarray(values, dtype=np.float64).copy()
    positive = sigma[(sigma > 0.0) & np.isfinite(sigma)]

    if positive.size == 0:
        fallback = 1e-12
    else:
        fallback = max(1e-12, float(np.median(positive)))

    invalid = (~np.isfinite(sigma)) | (sigma <= 0.0)
    sigma[invalid] = fallback
    return sigma


def fit_gap_cosh(
    tau: ArrayLike,
    G: ArrayLike,
    G_std: ArrayLike,
    *,
    beta: float,
    fit_window: tuple[int, int] | tuple[float, float] | None = None,
    initial_guess: tuple[float, float] | None = None,
) -> GapFitResult:
    """Fit correlator data to a finite-temperature cosh model.

    Model:
        f(tau; A, Delta) = A * [exp(-Delta*tau) + exp(-Delta*(beta-tau))]

    The fit is weighted using `sigma=G_std` in the selected fit window.
    """
    tau_arr = np.asarray(tau, dtype=np.float64)
    g_arr = np.asarray(G, dtype=np.float64)
    err_arr = _safe_sigma(np.asarray(G_std, dtype=np.float64))

    if tau_arr.ndim != 1 or g_arr.ndim != 1 or err_arr.ndim != 1:
        raise ValueError("tau, G, and G_std must be one-dimensional")
    if not (tau_arr.size == g_arr.size == err_arr.size):
        raise ValueError("tau, G, and G_std must have the same length")

    indices = _resolve_fit_indices(tau_arr, beta=beta, fit_window=fit_window)
    x = tau_arr[indices]
    y = g_arr[indices]
    sigma = err_arr[indices]

    if initial_guess is None:
        amplitude0 = max(float(np.max(y)), 1e-12)
        if x.size >= 2 and y[0] > 0.0 and y[-1] > 0.0 and (x[-1] > x[0]):
            delta0 = np.log(y[0] / y[-1]) / (x[-1] - x[0])
            delta0 = max(float(delta0), 1e-6)
        else:
            delta0 = max(1e-6, 2.0 / beta)
    else:
        amplitude0, delta0 = initial_guess

    try:
        params, covariance = curve_fit(
            lambda t, amplitude, delta: _cosh_model(t, amplitude, delta, beta),
            x,
            y,
            p0=(amplitude0, delta0),
            sigma=sigma,
            absolute_sigma=True,
            bounds=((1e-14, 1e-10), (np.inf, np.inf)),
            maxfev=20_000,
        )
    except Exception as exc:  # pragma: no cover - exercised indirectly
        raise ValueError(f"cosh gap fit failed: {exc}") from exc

    amp = float(params[0])
    delta = float(params[1])

    amp_var = float(covariance[0, 0]) if covariance.size else np.nan
    delta_var = float(covariance[1, 1]) if covariance.size else np.nan

    amp_std = float(np.sqrt(max(amp_var, 0.0))) if np.isfinite(amp_var) else np.nan
    delta_std = (
        float(np.sqrt(max(delta_var, 0.0))) if np.isfinite(delta_var) else np.nan
    )

    return GapFitResult(
        amplitude=amp,
        amplitude_std=amp_std,
        delta=delta,
        delta_std=delta_std,
        covariance=np.asarray(covariance, dtype=np.float64),
        fit_indices=indices,
    )


def bootstrap_gap_from_correlators(
    bootstrap_correlators: ArrayLike,
    tau: ArrayLike,
    *,
    beta: float,
    G_std: ArrayLike,
    fit_window: tuple[int, int] | tuple[float, float] | None = None,
) -> GapBootstrapResult:
    """Bootstrap a gap estimate by fitting each replicate correlator."""
    boot = np.asarray(bootstrap_correlators, dtype=np.float64)
    tau_arr = np.asarray(tau, dtype=np.float64)
    err_arr = np.asarray(G_std, dtype=np.float64)

    if boot.ndim != 2:
        raise ValueError("bootstrap_correlators must be a 2D array")
    if boot.shape[1] != tau_arr.size:
        raise ValueError("bootstrap_correlators and tau have incompatible shapes")

    deltas: list[float] = []
    for replicate in boot:
        try:
            fit = fit_gap_cosh(
                tau=tau_arr,
                G=replicate,
                G_std=err_arr,
                beta=beta,
                fit_window=fit_window,
            )
        except ValueError:
            continue
        if np.isfinite(fit.delta) and fit.delta > 0.0:
            deltas.append(fit.delta)

    if not deltas:
        raise ValueError("no successful bootstrap fits for the gap")

    delta_arr = np.asarray(deltas, dtype=np.float64)
    delta_mean = float(np.mean(delta_arr))
    delta_std = (
        float(np.std(delta_arr, ddof=1)) if delta_arr.size > 1 else 0.0
    )
    q16, q84 = np.percentile(delta_arr, [16.0, 84.0])

    return GapBootstrapResult(
        delta_mean=delta_mean,
        delta_std=delta_std,
        delta_q16=float(q16),
        delta_q84=float(q84),
        deltas=delta_arr,
        n_success=int(delta_arr.size),
    )


def integrated_autocorrelation_time(
    series: ArrayLike,
    max_lag: int | None = None,
) -> float:
    """Compute a simple integrated autocorrelation-time estimate."""
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional")
    if values.size < 2:
        return 0.0

    centered = values - float(np.mean(values))
    variance = float(np.var(centered))
    if variance == 0.0:
        return 0.0

    n = centered.size
    if max_lag is None:
        max_lag = min(n // 2, 1000)
    max_lag = min(max_lag, n - 1)

    autocorr = np.empty(max_lag + 1, dtype=np.float64)
    autocorr[0] = 1.0

    for lag in range(1, max_lag + 1):
        covariance = float(np.dot(centered[:-lag], centered[lag:])) / float(n - lag)
        autocorr[lag] = covariance / variance

    positive_lags = autocorr[1:]
    positive_lags = positive_lags[positive_lags > 0.0]
    return float(0.5 + np.sum(positive_lags))


def bootstrap_mean(
    samples: ArrayLike,
    n_resamples: int = 300,
    seed: int = 12345,
) -> tuple[float, float]:
    """Estimate mean and bootstrap standard error for 1D samples."""
    values = np.asarray(samples, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("samples must contain at least one element")
    if n_resamples < 2:
        raise ValueError("n_resamples must be >= 2")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(
        0,
        values.size,
        size=(n_resamples, values.size),
        endpoint=False,
    )
    resampled_means = np.mean(values[indices], axis=1)

    mean_estimate = float(np.mean(resampled_means))
    error_estimate = float(np.std(resampled_means, ddof=1))
    return mean_estimate, error_estimate
