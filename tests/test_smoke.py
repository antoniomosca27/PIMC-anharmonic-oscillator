"""Fast deterministic smoke test for the core analysis workflow."""

from __future__ import annotations

import numpy as np
from pimc_oscillator import LatticeParams, PotentialParams, RunConfig, SamplerParams
from pimc_oscillator.analysis import effective_mass_from_correlator, fit_gap_cosh
from pimc_oscillator.observables import bootstrap_correlator
from pimc_oscillator.sampler import run_pimc


def test_smoke_small_end_to_end_pipeline() -> None:
    """Run a tiny deterministic pipeline from sampling to gap extraction."""
    config = RunConfig(
        potential=PotentialParams(m=1.0, omega=1.0, lam=0.25),
        lattice=LatticeParams(beta=3.0, n_slices=24),
        sampler=SamplerParams(
            n_therm=20,
            n_sweeps=60,
            measure_every=6,
            proposal_width=0.8,
            tune=True,
            tune_interval=10,
            target_accept=0.5,
            seed=31415,
        ),
        run_name="smoke",
    )

    chain = run_pimc(config=config, progress=False)
    corr = bootstrap_correlator(
        chain.samples,
        subtract_mean=False,
        bin_size=5,
        n_boot=40,
        seed=31416,
    )

    tau = np.arange(config.lattice.n_slices, dtype=np.float64) * config.a
    tau_eff, delta_eff = effective_mass_from_correlator(corr.G_mean, a=config.a)
    fit = fit_gap_cosh(
        tau=tau,
        G=corr.G_mean,
        G_std=corr.G_std,
        beta=config.lattice.beta,
        fit_window=(5, 11),
    )

    assert chain.samples.shape == (config.n_samples, config.lattice.n_slices)
    assert corr.G_mean.shape == (config.lattice.n_slices,)
    assert corr.G_std.shape == (config.lattice.n_slices,)
    assert tau_eff.shape == delta_eff.shape
    assert np.all(np.isfinite(delta_eff))
    assert np.isfinite(fit.delta)
    assert fit.delta > 0.0
