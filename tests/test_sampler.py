"""Tests for sampler invariants and deterministic behavior."""

from __future__ import annotations

import numpy as np
import pytest

from pimc_oscillator import (
    LatticeParams,
    PotentialParams,
    RunConfig,
    SamplerParams,
    run_pimc,
)


def _small_sampler_config() -> RunConfig:
    """Build a compact deterministic run configuration for sampler tests."""
    return RunConfig(
        potential=PotentialParams(m=1.0, omega=1.0, lam=0.4),
        lattice=LatticeParams(beta=4.0, n_slices=32),
        sampler=SamplerParams(
            n_therm=50,
            n_sweeps=200,
            measure_every=10,
            proposal_width=0.9,
            tune=True,
            tune_interval=10,
            target_accept=0.5,
            seed=2024,
        ),
        run_name="sampler-test",
    )


def _small_hmc_config() -> RunConfig:
    """Build a compact deterministic run configuration for HMC tests."""
    return RunConfig(
        potential=PotentialParams(m=1.0, omega=1.0, lam=0.4),
        lattice=LatticeParams(beta=4.0, n_slices=24),
        sampler=SamplerParams(
            method="hmc",
            n_therm=15,
            n_sweeps=50,
            measure_every=5,
            proposal_width=1.0,
            tune=True,
            tune_interval=5,
            target_accept=0.75,
            hmc_step_size=0.045,
            hmc_n_leapfrog=16,
            hmc_mass=1.0,
            seed=77,
        ),
        run_name="hmc-test",
    )


def test_sampler_shape_and_metadata_invariants() -> None:
    """Sampler should return correctly shaped samples and metadata fields."""
    config = _small_sampler_config()
    chain = run_pimc(config=config, progress=False)

    assert chain.samples.shape == (config.n_samples, config.lattice.n_slices)
    assert 0.0 <= chain.acceptance_rate <= 1.0
    assert chain.final_proposal_width > 0.0
    assert chain.final_control_value == chain.final_proposal_width

    assert chain.metadata["beta"] == config.lattice.beta
    assert chain.metadata["n_slices"] == config.lattice.n_slices
    assert np.isclose(chain.metadata["a"], config.a)
    assert chain.metadata["seed"] == config.sampler.seed
    assert chain.metadata["final_control_name"] == "proposal_width"
    assert np.isclose(chain.metadata["final_control_value"], chain.final_control_value)


def test_sampler_is_deterministic_for_fixed_seed() -> None:
    """Running the same configuration twice should produce identical outputs."""
    config = _small_sampler_config()

    chain_a = run_pimc(config=config, progress=False)
    chain_b = run_pimc(config=config, progress=False)

    assert np.array_equal(chain_a.samples, chain_b.samples)
    assert chain_a.acceptance_rate == chain_b.acceptance_rate
    assert chain_a.final_proposal_width == chain_b.final_proposal_width


def test_metropolis_rb_requires_even_n_slices() -> None:
    """Red-black vectorized updates with periodic boundaries require even N."""
    config = RunConfig(
        potential=PotentialParams(m=1.0, omega=1.0, lam=0.2),
        lattice=LatticeParams(beta=3.0, n_slices=25),
        sampler=SamplerParams(
            method="metropolis_rb",
            n_therm=5,
            n_sweeps=10,
            measure_every=2,
            proposal_width=0.7,
            tune=False,
            seed=1,
        ),
        run_name="odd-metropolis",
    )

    with pytest.raises(ValueError, match="requires an even n_slices"):
        run_pimc(config=config, progress=False)


def test_hmc_is_deterministic_for_fixed_seed() -> None:
    """HMC backend must be deterministic under identical run configs."""
    config = _small_hmc_config()

    chain_a = run_pimc(config=config, progress=False)
    chain_b = run_pimc(config=config, progress=False)

    assert np.array_equal(chain_a.samples, chain_b.samples)
    assert chain_a.acceptance_rate == chain_b.acceptance_rate
    assert chain_a.final_proposal_width == chain_b.final_proposal_width
    assert chain_a.final_control_value == chain_a.final_proposal_width
    assert chain_a.metadata["method"] == "hmc"
    assert chain_a.metadata["final_control_name"] == "hmc_step_size"
    assert np.isclose(
        chain_a.metadata["final_control_value"],
        chain_a.final_control_value,
    )
