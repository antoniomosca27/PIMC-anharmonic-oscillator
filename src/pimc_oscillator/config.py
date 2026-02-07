"""Structured configuration objects for PIMC anharmonic-oscillator runs.

The dataclasses in this module define the canonical runtime inputs for the
simulation engine. Validation is eager and strict so that invalid parameter
choices fail fast at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal


@dataclass(frozen=True, slots=True)
class PotentialParams:
    """Physical parameters for the 1D quartic anharmonic oscillator.

    The Hamiltonian convention is:
        H = p^2 / (2m) + V(x)
        V(x) = 0.5 * m * omega^2 * x^2 + lam * x^4

    Attributes:
        m: Mass parameter, constrained to strictly positive values.
        omega: Harmonic frequency, constrained to non-negative values.
        lam: Quartic coupling, constrained to non-negative values.
    """

    m: float = 1.0
    omega: float = 1.0
    lam: float = 1.0

    def __post_init__(self) -> None:
        """Validate physical constraints for the potential model."""
        if not isfinite(self.m) or self.m <= 0.0:
            raise ValueError("m must be a finite strictly positive float")
        if not isfinite(self.omega) or self.omega < 0.0:
            raise ValueError("omega must be a finite non-negative float")
        if not isfinite(self.lam) or self.lam < 0.0:
            raise ValueError("lam must be a finite non-negative float")


@dataclass(frozen=True, slots=True)
class LatticeParams:
    """Euclidean-time lattice parameters.

    Attributes:
        beta: Inverse temperature, constrained to strictly positive values.
        n_slices: Number of Euclidean time slices N, constrained to N >= 4.
    """

    beta: float
    n_slices: int

    def __post_init__(self) -> None:
        """Validate lattice geometry constraints."""
        if not isfinite(self.beta) or self.beta <= 0.0:
            raise ValueError("beta must be a finite strictly positive float")
        if self.n_slices < 4:
            raise ValueError("n_slices must be an integer >= 4")

    @property
    def a(self) -> float:
        """Return the Euclidean lattice spacing a = beta / N."""
        return self.beta / float(self.n_slices)


@dataclass(frozen=True, slots=True)
class SamplerParams:
    """Sampler controls for Metropolis red-black and HMC backends.

    Attributes:
        method: Sampling backend (`"metropolis_rb"` or `"hmc"`).
        n_therm: Number of thermalization sweeps.
        n_sweeps: Number of production sweeps.
        measure_every: Measurement stride in production sweeps.
        proposal_width: Uniform proposal half-width (Metropolis backend).
        tune: Whether proposal-width tuning is enabled during thermalization.
        tune_interval: Sweep interval used by the tuning controller.
        target_accept: Target acceptance fraction for tuning.
        hmc_step_size: Leapfrog step size (HMC backend).
        hmc_n_leapfrog: Leapfrog sub-steps per HMC proposal.
        hmc_mass: Momentum mass parameter used in HMC kinetic term.
        seed: Deterministic random seed.
    """

    method: Literal["metropolis_rb", "hmc"] = "metropolis_rb"
    n_therm: int = 500
    n_sweeps: int = 1_000
    measure_every: int = 10
    proposal_width: float = 1.0
    tune: bool = True
    tune_interval: int = 50
    target_accept: float = 0.5
    hmc_step_size: float = 0.05
    hmc_n_leapfrog: int = 20
    hmc_mass: float = 1.0
    seed: int = 12345

    def __post_init__(self) -> None:
        """Validate sampler hyper-parameters."""
        if self.method not in {"metropolis_rb", "hmc"}:
            raise ValueError("method must be 'metropolis_rb' or 'hmc'")
        if self.n_therm < 0:
            raise ValueError("n_therm must be >= 0")
        if self.n_sweeps < 1:
            raise ValueError("n_sweeps must be >= 1")
        if self.measure_every < 1:
            raise ValueError("measure_every must be >= 1")
        if not isfinite(self.proposal_width) or self.proposal_width <= 0.0:
            raise ValueError("proposal_width must be a finite strictly positive float")
        if self.tune_interval < 1:
            raise ValueError("tune_interval must be >= 1")
        if not isfinite(self.target_accept) or not 0.0 < self.target_accept < 1.0:
            raise ValueError("target_accept must satisfy 0 < target_accept < 1")
        if not isfinite(self.hmc_step_size) or self.hmc_step_size <= 0.0:
            raise ValueError("hmc_step_size must be a finite strictly positive float")
        if self.hmc_n_leapfrog < 1:
            raise ValueError("hmc_n_leapfrog must be >= 1")
        if not isfinite(self.hmc_mass) or self.hmc_mass <= 0.0:
            raise ValueError("hmc_mass must be a finite strictly positive float")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Complete immutable configuration for one PIMC chain run."""

    potential: PotentialParams
    lattice: LatticeParams
    sampler: SamplerParams
    run_name: str | None = None

    def __post_init__(self) -> None:
        """Validate run-level metadata constraints."""
        if self.run_name is not None and self.run_name.strip() == "":
            raise ValueError("run_name cannot be an empty string")

    @property
    def a(self) -> float:
        """Convenience proxy for lattice spacing."""
        return self.lattice.a

    @property
    def n_samples(self) -> int:
        """Number of stored configurations implied by production controls."""
        return (self.sampler.n_sweeps - 1) // self.sampler.measure_every + 1

    @property
    def total_sweeps(self) -> int:
        """Total number of sweeps including thermalization."""
        return self.sampler.n_therm + self.sampler.n_sweeps
