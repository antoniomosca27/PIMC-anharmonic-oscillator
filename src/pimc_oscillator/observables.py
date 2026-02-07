"""Observable estimators for sampled PIMC paths.

This module provides fast FFT-based estimators for translationally averaged
Euclidean correlators and uncertainty estimation via binning + bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CorrelatorEstimate:
    """Correlator estimate with bootstrap uncertainty information.

    Attributes:
        G_mean: Mean correlator over configurations.
        G_std: Bootstrap standard deviation of the correlator.
        G_q16: Optional 16th percentile band from bootstrap samples.
        G_q84: Optional 84th percentile band from bootstrap samples.
        bootstrap_samples: Optional bootstrap replicate correlators.
        n_bins: Number of bins used in the bootstrap stage.
        bin_size: Bin size used in the binning stage.
        subtract_mean: Whether per-configuration mean subtraction was applied.
    """

    G_mean: FloatArray
    G_std: FloatArray
    G_q16: FloatArray | None
    G_q84: FloatArray | None
    bootstrap_samples: FloatArray | None
    n_bins: int
    bin_size: int
    subtract_mean: bool


def _validate_paths(paths: ArrayLike) -> FloatArray:
    """Validate and convert sampled paths to a 2D float64 array."""
    arr = np.asarray(paths, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("paths must be a 2D array with shape (n_samples, n_slices)")
    if arr.shape[0] < 1:
        raise ValueError("paths must contain at least one configuration")
    if arr.shape[1] < 4:
        raise ValueError("paths must contain at least 4 lattice slices")
    return arr


def mean_position(paths: ArrayLike) -> float:
    """Estimate <x> from sampled paths."""
    arr = _validate_paths(paths)
    return float(np.mean(arr))


def mean_square_position(paths: ArrayLike) -> float:
    """Estimate <x^2> from sampled paths."""
    arr = _validate_paths(paths)
    return float(np.mean(arr**2))


def correlator_single_configuration(
    configuration: ArrayLike,
    *,
    subtract_mean: bool = False,
) -> FloatArray:
    """Compute translationally averaged correlator g_k(x) for one path.

    For a single configuration `x` with N slices, this computes:
        g_k(x) = (1/N) * sum_i x_i * x_{i+k mod N}

    The implementation uses the FFT identity:
        ifft(fft(x) * conj(fft(x))).real

    Args:
        configuration: 1D configuration array of length N.
        subtract_mean: If True, use `x - mean(x)` before autocorrelation.

    Returns:
        Correlator array `g` with length N.
    """
    x = np.asarray(configuration, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("configuration must be one-dimensional")
    if x.size < 4:
        raise ValueError("configuration must have at least 4 entries")

    if subtract_mean:
        x = x - float(np.mean(x))

    fft_x = np.fft.fft(x)
    corr = np.fft.ifft(fft_x * np.conjugate(fft_x)).real
    return corr / float(x.size)


def correlator_per_configuration(
    paths: ArrayLike,
    *,
    subtract_mean: bool = False,
) -> FloatArray:
    """Compute translationally averaged correlators for each configuration.

    Args:
        paths: Array with shape `(n_samples, n_slices)`.
        subtract_mean: If True, subtract per-configuration means before FFT.

    Returns:
        Array with shape `(n_samples, n_slices)` where each row is `g_k(x)`.
    """
    arr = _validate_paths(paths)

    work = arr
    if subtract_mean:
        work = arr - np.mean(arr, axis=1, keepdims=True)

    fft_work = np.fft.fft(work, axis=1)
    corr = np.fft.ifft(fft_work * np.conjugate(fft_work), axis=1).real
    return corr / float(arr.shape[1])


def translational_correlator(
    paths: ArrayLike,
    *,
    subtract_mean: bool = False,
) -> FloatArray:
    """Estimate the ensemble correlator G_k from sampled configurations."""
    per_configuration = correlator_per_configuration(paths, subtract_mean=subtract_mean)
    return np.mean(per_configuration, axis=0)


def _bin_observables(observables: FloatArray, bin_size: int) -> tuple[FloatArray, int]:
    """Bin consecutive observable rows and return binned means.

    Args:
        observables: Array with shape `(n_samples, n_observables)`.
        bin_size: Number of consecutive samples per bin.

    Returns:
        Tuple `(binned, used_bin_size)` where `binned` has shape
        `(n_bins, n_observables)`.
    """
    if observables.ndim != 2:
        raise ValueError("observables must be a 2D array")
    if bin_size < 1:
        raise ValueError("bin_size must be >= 1")

    n_samples = observables.shape[0]
    if n_samples < bin_size:
        # Fall back to one-sample bins instead of dropping all data.
        return observables.copy(), 1

    n_bins = n_samples // bin_size
    trimmed = observables[: n_bins * bin_size]
    binned = trimmed.reshape(n_bins, bin_size, observables.shape[1]).mean(axis=1)
    return binned, bin_size


def bootstrap_correlator(
    paths: ArrayLike,
    *,
    subtract_mean: bool = False,
    bin_size: int = 10,
    n_boot: int = 300,
    seed: int = 12345,
    return_quantiles: bool = True,
    return_bootstrap_samples: bool = False,
) -> CorrelatorEstimate:
    """Estimate correlator mean and error bars via binning + bootstrap.

    Pipeline:
    1. Compute FFT-based `g_k(x)` for each configuration.
    2. Bin consecutive rows into bins of size `bin_size`.
    3. Bootstrap resample bins with replacement `n_boot` times.

    Args:
        paths: Sampled configurations with shape `(n_samples, n_slices)`.
        subtract_mean: Whether to subtract each configuration mean before FFT.
        bin_size: Number of consecutive configurations per bin.
        n_boot: Number of bootstrap replicates.
        seed: Bootstrap RNG seed.
        return_quantiles: If True, compute 16/84 bootstrap bands.
        return_bootstrap_samples: If True, include bootstrap replicas.

    Returns:
        `CorrelatorEstimate` containing mean, errors, and metadata.
    """
    if n_boot < 2:
        raise ValueError("n_boot must be >= 2")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    g_cfg = correlator_per_configuration(paths, subtract_mean=subtract_mean)
    g_binned, used_bin_size = _bin_observables(g_cfg, bin_size=bin_size)

    n_bins = g_binned.shape[0]
    rng = np.random.Generator(np.random.PCG64(seed))

    if n_bins == 1:
        boot = np.repeat(g_binned, repeats=n_boot, axis=0)
    else:
        indices = rng.integers(0, n_bins, size=(n_boot, n_bins), endpoint=False)
        boot = np.mean(g_binned[indices], axis=1)

    g_mean = np.mean(g_binned, axis=0)
    g_std = np.std(boot, axis=0, ddof=1)

    g_q16: FloatArray | None = None
    g_q84: FloatArray | None = None
    if return_quantiles:
        g_q16, g_q84 = np.percentile(boot, [16.0, 84.0], axis=0)

    boot_payload: FloatArray | None = None
    if return_bootstrap_samples:
        boot_payload = boot

    return CorrelatorEstimate(
        G_mean=g_mean,
        G_std=g_std,
        G_q16=g_q16,
        G_q84=g_q84,
        bootstrap_samples=boot_payload,
        n_bins=n_bins,
        bin_size=used_bin_size,
        subtract_mean=subtract_mean,
    )


def two_point_correlation(
    paths: ArrayLike,
    max_lag: int,
    *,
    subtract_mean: bool = False,
) -> FloatArray:
    """Convenience helper for two-point correlator values.

    Args:
        paths: Sampled configurations with shape `(n_samples, n_slices)`.
        max_lag: Maximum lag index to include.
        subtract_mean: Whether to subtract each configuration mean before FFT.

    Returns:
        Correlator values `G_0 ... G_max_lag`.
    """
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")

    full = translational_correlator(paths, subtract_mean=subtract_mean)
    lag_limit = min(max_lag, full.size - 1)
    return full[: lag_limit + 1].copy()
