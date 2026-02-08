"""High-level experiment orchestration for PIMC production runs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.analysis import (
    bootstrap_gap_from_correlators,
    bootstrap_mean,
    default_fit_window_indices,
    effective_mass_bootstrap_from_correlators,
    effective_mass_from_correlator,
    fit_gap_cosh,
    harmonic_correlator,
    harmonic_x2,
    integrated_autocorrelation_time,
    symmetrize_periodic_correlator,
)
from src.config import RunConfig
from src.io import (
    RunPaths,
    create_run_directories,
    save_analysis_json,
    save_chain_npz,
    save_config_json,
    save_run_summary_json,
)
from src.observables import (
    bootstrap_correlator,
    mean_position,
    mean_square_position,
)
from src.sampler import run_pimc

FloatArray = NDArray[np.float64]


def _lam_tag(lam: float) -> str:
    return f"lam_{lam:.2f}".replace(".", "p")


def _prefixed_name(prefix: str | None, suffix: str) -> str:
    if prefix is None or prefix == "":
        return suffix
    return f"{prefix}_{suffix}"


def _symmetrize_bootstrap_correlators(boot: FloatArray) -> FloatArray:
    mirrored = boot[:, (-np.arange(boot.shape[1])) % boot.shape[1]]
    return 0.5 * (boot + mirrored)


def _effective_sample_size(n_samples: int, tau_int: float) -> float:
    if tau_int <= 0.0:
        return float(n_samples)
    return float(n_samples) / max(1.0, 2.0 * tau_int)


def _recommended_bin_size(n_samples: int, tau_int: float) -> int:
    if tau_int <= 0.0:
        return 1
    return max(1, min(n_samples, int(np.ceil(2.0 * tau_int))))


def run_single_case(
    config: RunConfig,
    *,
    out_root: str | Path = ".",
    run_paths: RunPaths | None = None,
    artifact_prefix: str | None = None,
    bin_size: int = 10,
    n_boot: int = 300,
    subtract_mean: bool = False,
    fit_tau_min_frac: float = 0.2,
    fit_tau_max_frac: float = 0.5,
    symmetrize_correlator: bool = False,
    effective_mass_invalid_as_nan: bool = False,
    effective_mass_snr_min: float | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Run one coupling point and persist chain + analysis artifacts.

    Args:
        config: Complete immutable run configuration.
        out_root: Output root used when run_paths is not supplied.
        run_paths: Optional pre-created shared run directories.
        artifact_prefix: Optional filename prefix (for lambda scans).
        bin_size: Correlator bootstrap bin size.
        n_boot: Number of bootstrap replicates.
        subtract_mean: Whether to subtract per-path means in correlators.
        fit_tau_min_frac: Lower fit-window fraction of beta.
        fit_tau_max_frac: Upper fit-window fraction of beta.
        symmetrize_correlator: Use G_sym for effective mass and cosh fits.
        effective_mass_invalid_as_nan: Return NaN for invalid Delta_eff points.
        effective_mass_snr_min: Optional SNR mask threshold for Delta_eff.
        progress: Show sampler progress bar.

    Returns:
        Notebook-friendly dictionary with arrays, fit summaries, diagnostics,
        and all persisted artifact paths.
    """
    if bin_size < 1:
        raise ValueError("bin_size must be >= 1")
    if n_boot < 2:
        raise ValueError("n_boot must be >= 2")

    if run_paths is None:
        run_paths = create_run_directories(out_root=out_root, run_name=config.run_name)

    chain = run_pimc(config=config, progress=progress)
    samples = chain.samples
    if samples.shape[0] < 1:
        raise ValueError("sampler produced no stored samples")
    final_control_name = (
        "proposal_width"
        if config.sampler.method == "metropolis_rb"
        else "hmc_step_size"
    )

    n_slices = config.lattice.n_slices
    tau = np.arange(n_slices, dtype=np.float64) * config.a

    correlator = bootstrap_correlator(
        samples,
        subtract_mean=subtract_mean,
        bin_size=bin_size,
        n_boot=n_boot,
        seed=config.sampler.seed + 11,
        return_quantiles=True,
        return_bootstrap_samples=True,
    )

    if correlator.bootstrap_samples is None:
        raise RuntimeError("bootstrap correlator samples were not retained")

    g_for_analysis = correlator.G_mean
    g_std_for_analysis = correlator.G_std
    g_boot_for_analysis = correlator.bootstrap_samples

    if symmetrize_correlator:
        g_for_analysis = symmetrize_periodic_correlator(g_for_analysis)
        g_std_for_analysis = symmetrize_periodic_correlator(g_std_for_analysis)
        g_boot_for_analysis = _symmetrize_bootstrap_correlators(g_boot_for_analysis)

    tau_eff, delta_eff = effective_mass_from_correlator(
        g_for_analysis,
        a=config.a,
        invalid_as_nan=effective_mass_invalid_as_nan,
        G_err=g_std_for_analysis,
        snr_min=effective_mass_snr_min,
    )
    eff_mass_boot = effective_mass_bootstrap_from_correlators(
        g_boot_for_analysis,
        a=config.a,
        invalid_as_nan=effective_mass_invalid_as_nan,
        G_err=g_std_for_analysis,
        snr_min=effective_mass_snr_min,
    )

    k_min, k_max = default_fit_window_indices(
        beta=config.lattice.beta,
        a=config.a,
        n_slices=config.lattice.n_slices,
        tau_min_frac=fit_tau_min_frac,
        tau_max_frac=fit_tau_max_frac,
    )

    fit_result = fit_gap_cosh(
        tau=tau,
        G=g_for_analysis,
        G_std=g_std_for_analysis,
        beta=config.lattice.beta,
        fit_window=(k_min, k_max),
    )

    gap_boot = bootstrap_gap_from_correlators(
        bootstrap_correlators=g_boot_for_analysis,
        tau=tau,
        beta=config.lattice.beta,
        G_std=g_std_for_analysis,
        fit_window=(k_min, k_max),
    )

    x_mean = mean_position(samples)
    x2_mean = mean_square_position(samples)
    x2_per_configuration = np.mean(samples**2, axis=1)
    x2_boot_mean, x2_boot_std = bootstrap_mean(
        x2_per_configuration,
        n_resamples=max(200, n_boot),
        seed=config.sampler.seed + 29,
    )

    tau_int_x2 = integrated_autocorrelation_time(x2_per_configuration)
    ess_x2 = _effective_sample_size(samples.shape[0], tau_int_x2)
    suggested_bin_size = _recommended_bin_size(samples.shape[0], tau_int_x2)

    g_exact: np.ndarray | None = None
    x2_exact: float | None = None
    if config.potential.lam == 0.0 and config.potential.omega > 0.0:
        g_exact = harmonic_correlator(
            tau=tau,
            beta=config.lattice.beta,
            m=config.potential.m,
            omega=config.potential.omega,
        )
        x2_exact = harmonic_x2(
            beta=config.lattice.beta,
            m=config.potential.m,
            omega=config.potential.omega,
        )

    config_path = save_config_json(
        config,
        run_paths.logs_dir / _prefixed_name(artifact_prefix, "config.json"),
    )
    chain_path = save_chain_npz(
        samples=samples,
        metadata=chain.metadata,
        path=run_paths.logs_dir / _prefixed_name(artifact_prefix, "chain.npz"),
    )

    analysis_payload: dict[str, Any] = {
        "run_id": run_paths.run_id,
        "artifact_prefix": artifact_prefix,
        "sampler_method": config.sampler.method,
        "acceptance_rate": chain.acceptance_rate,
        "final_control_name": final_control_name,
        "final_control_value": chain.final_control_value,
        "final_proposal_width": chain.final_proposal_width,
        "x_mean": x_mean,
        "x2_mean": x2_mean,
        "x2_bootstrap_mean": x2_boot_mean,
        "x2_bootstrap_std": x2_boot_std,
        "diagnostics": {
            "tau_int_x2": tau_int_x2,
            "effective_sample_size_x2": ess_x2,
            "n_stored_samples": int(samples.shape[0]),
            "suggested_bin_size": suggested_bin_size,
        },
        "correlator": {
            "subtract_mean": correlator.subtract_mean,
            "symmetrized": symmetrize_correlator,
            "bin_size": correlator.bin_size,
            "n_bins": correlator.n_bins,
            "tau": tau.tolist(),
            "G": correlator.G_mean.tolist(),
            "G_err": correlator.G_std.tolist(),
            "G_analysis": g_for_analysis.tolist(),
            "G_err_analysis": g_std_for_analysis.tolist(),
            "G_q16": None if correlator.G_q16 is None else correlator.G_q16.tolist(),
            "G_q84": None if correlator.G_q84 is None else correlator.G_q84.tolist(),
        },
        "effective_mass": {
            "tau": tau_eff.tolist(),
            "Delta_eff": delta_eff.tolist(),
            "invalid_as_nan": effective_mass_invalid_as_nan,
            "snr_min": effective_mass_snr_min,
        },
        "effective_mass_bootstrap": {
            "tau": eff_mass_boot.tau_mid.tolist(),
            "mean": eff_mass_boot.mean.tolist(),
            "std": eff_mass_boot.std.tolist(),
            "q16": eff_mass_boot.q16.tolist(),
            "q84": eff_mass_boot.q84.tolist(),
            "n_valid": eff_mass_boot.n_valid.tolist(),
            "invalid_as_nan": effective_mass_invalid_as_nan,
            "snr_min": effective_mass_snr_min,
        },
        "gap_fit": {
            "amplitude": fit_result.amplitude,
            "amplitude_std": fit_result.amplitude_std,
            "delta": fit_result.delta,
            "delta_std": fit_result.delta_std,
            "fit_indices": fit_result.fit_indices.tolist(),
            "fit_tau_min": float(tau[k_min]),
            "fit_tau_max": float(tau[k_max]),
        },
        "gap_bootstrap": {
            "delta_mean": gap_boot.delta_mean,
            "delta_std": gap_boot.delta_std,
            "delta_q16": gap_boot.delta_q16,
            "delta_q84": gap_boot.delta_q84,
            "n_success": gap_boot.n_success,
        },
    }

    if g_exact is not None:
        analysis_payload["harmonic_exact"] = {
            "G_exact": g_exact.tolist(),
            "x2_exact": x2_exact,
        }

    analysis_path = save_analysis_json(
        analysis_payload,
        run_paths.reports_dir / _prefixed_name(artifact_prefix, "analysis.json"),
    )

    summary_payload: dict[str, Any] = {
        "run_id": run_paths.run_id,
        "run_name": config.run_name,
        "artifact_prefix": artifact_prefix,
        "sampler_method": config.sampler.method,
        "acceptance_rate": chain.acceptance_rate,
        "final_control_name": final_control_name,
        "final_control_value": chain.final_control_value,
        "final_proposal_width": chain.final_proposal_width,
        "delta_fit": fit_result.delta,
        "delta_fit_std": fit_result.delta_std,
        "delta_bootstrap_mean": gap_boot.delta_mean,
        "delta_bootstrap_std": gap_boot.delta_std,
        "tau_int_x2": tau_int_x2,
        "effective_sample_size_x2": ess_x2,
        "suggested_bin_size": suggested_bin_size,
        "logs_dir": str(run_paths.logs_dir),
        "reports_dir": str(run_paths.reports_dir),
        "config_path": str(config_path),
        "chain_path": str(chain_path),
        "analysis_path": str(analysis_path),
    }
    summary_path = save_run_summary_json(
        summary_payload,
        run_paths.reports_dir / _prefixed_name(artifact_prefix, "run_summary.json"),
    )

    return {
        "run_id": run_paths.run_id,
        "paths": {
            "logs_dir": run_paths.logs_dir,
            "reports_dir": run_paths.reports_dir,
            "config_path": config_path,
            "chain_path": chain_path,
            "analysis_path": analysis_path,
            "run_summary_path": summary_path,
        },
        "config": config,
        "chain": chain,
        "samples": samples,
        "tau": tau,
        "G": g_for_analysis,
        "G_err": g_std_for_analysis,
        "G_raw": correlator.G_mean,
        "G_err_raw": correlator.G_std,
        "G_q16": correlator.G_q16,
        "G_q84": correlator.G_q84,
        "effective_mass_tau": tau_eff,
        "effective_mass": delta_eff,
        "effective_mass_bootstrap": eff_mass_boot,
        "fit_window_indices": np.arange(k_min, k_max + 1, dtype=np.int64),
        "fit_window_tau": (float(tau[k_min]), float(tau[k_max])),
        "gap_fit": fit_result,
        "gap_bootstrap": gap_boot,
        "acceptance_rate": chain.acceptance_rate,
        "final_control_name": final_control_name,
        "final_control_value": chain.final_control_value,
        "final_proposal_width": chain.final_proposal_width,
        "x_mean": x_mean,
        "x2_mean": x2_mean,
        "x2_bootstrap_mean": x2_boot_mean,
        "x2_bootstrap_std": x2_boot_std,
        "tau_int_x2": tau_int_x2,
        "effective_sample_size_x2": ess_x2,
        "suggested_bin_size": suggested_bin_size,
        "G_exact": g_exact,
        "x2_exact": x2_exact,
    }


def run_lambda_scan(
    lambdas: ArrayLike,
    *,
    base_config: RunConfig,
    out_root: str | Path = ".",
    run_name: str | None = None,
    seed_stride: int = 1000,
    bin_size: int = 10,
    n_boot: int = 300,
    subtract_mean: bool = False,
    fit_tau_min_frac: float = 0.2,
    fit_tau_max_frac: float = 0.5,
    symmetrize_correlator: bool = False,
    effective_mass_invalid_as_nan: bool = False,
    effective_mass_snr_min: float | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Run multiple couplings in one shared run directory tree."""
    lam_values = np.asarray(lambdas, dtype=np.float64)
    if lam_values.ndim != 1 or lam_values.size < 1:
        raise ValueError("lambdas must be a one-dimensional non-empty array")

    run_label = run_name if run_name is not None else base_config.run_name
    run_paths = create_run_directories(out_root=out_root, run_name=run_label)

    results_by_lambda: dict[float, dict[str, Any]] = {}
    summary_rows: list[dict[str, float]] = []

    for idx, lam in enumerate(lam_values):
        lam_value = float(lam)
        case_config = replace(
            base_config,
            potential=replace(base_config.potential, lam=lam_value),
            sampler=replace(
                base_config.sampler,
                seed=base_config.sampler.seed + idx * seed_stride,
            ),
            run_name=(run_label or "lambda-scan") + f"_lam_{lam_value:.2f}",
        )

        case_result = run_single_case(
            case_config,
            run_paths=run_paths,
            artifact_prefix=_lam_tag(lam_value),
            bin_size=bin_size,
            n_boot=n_boot,
            subtract_mean=subtract_mean,
            fit_tau_min_frac=fit_tau_min_frac,
            fit_tau_max_frac=fit_tau_max_frac,
            symmetrize_correlator=symmetrize_correlator,
            effective_mass_invalid_as_nan=effective_mass_invalid_as_nan,
            effective_mass_snr_min=effective_mass_snr_min,
            progress=progress,
        )
        results_by_lambda[lam_value] = case_result

        summary_rows.append(
            {
                "lambda": lam_value,
                "gap": float(case_result["gap_bootstrap"].delta_mean),
                "error": float(case_result["gap_bootstrap"].delta_std),
                "fit_gap": float(case_result["gap_fit"].delta),
                "acceptance": float(case_result["acceptance_rate"]),
                "tau_int_x2": float(case_result["tau_int_x2"]),
                "effective_sample_size_x2": float(
                    case_result["effective_sample_size_x2"]
                ),
            }
        )

    lambda_sorted = np.array(sorted(results_by_lambda.keys()), dtype=np.float64)
    gaps = np.array(
        [results_by_lambda[lam]["gap_bootstrap"].delta_mean for lam in lambda_sorted],
        dtype=np.float64,
    )
    gap_errs = np.array(
        [results_by_lambda[lam]["gap_bootstrap"].delta_std for lam in lambda_sorted],
        dtype=np.float64,
    )

    summary_payload: dict[str, Any] = {
        "run_id": run_paths.run_id,
        "run_name": run_label,
        "lambda_sorted": lambda_sorted.tolist(),
        "gaps": gaps.tolist(),
        "gap_errs": gap_errs.tolist(),
        "analysis_params": {
            "bin_size": bin_size,
            "n_boot": n_boot,
            "subtract_mean": subtract_mean,
            "fit_tau_min_frac": fit_tau_min_frac,
            "fit_tau_max_frac": fit_tau_max_frac,
            "symmetrize_correlator": symmetrize_correlator,
            "effective_mass_invalid_as_nan": effective_mass_invalid_as_nan,
            "effective_mass_snr_min": effective_mass_snr_min,
        },
        "summary_table": summary_rows,
        "reports_dir": str(run_paths.reports_dir),
        "logs_dir": str(run_paths.logs_dir),
    }
    summary_path = save_run_summary_json(
        summary_payload,
        run_paths.reports_dir / "run_summary.json",
    )

    return {
        "run_id": run_paths.run_id,
        "paths": {
            "logs_dir": run_paths.logs_dir,
            "reports_dir": run_paths.reports_dir,
            "run_summary_path": summary_path,
        },
        "results_by_lambda": results_by_lambda,
        "lambda_sorted": lambda_sorted,
        "gaps": gaps,
        "gap_errs": gap_errs,
        "summary_rows": summary_rows,
    }


def run_experiment(
    config: RunConfig,
    out_root: str | Path = ".",
    *,
    bin_size: int = 10,
    n_boot: int = 300,
    subtract_mean: bool = False,
    fit_tau_min_frac: float = 0.2,
    fit_tau_max_frac: float = 0.5,
    symmetrize_correlator: bool = False,
    effective_mass_invalid_as_nan: bool = False,
    effective_mass_snr_min: float | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Wrapper around `run_single_case`.

    This one-shot API used by scripts and tests exposes the main analysis
    controls as explicit keyword arguments.
    """
    return run_single_case(
        config=config,
        out_root=out_root,
        artifact_prefix=None,
        bin_size=bin_size,
        n_boot=n_boot,
        subtract_mean=subtract_mean,
        fit_tau_min_frac=fit_tau_min_frac,
        fit_tau_max_frac=fit_tau_max_frac,
        symmetrize_correlator=symmetrize_correlator,
        effective_mass_invalid_as_nan=effective_mass_invalid_as_nan,
        effective_mass_snr_min=effective_mass_snr_min,
        progress=progress,
    )
