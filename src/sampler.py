"""PIMC samplers with vectorized Metropolis red-black and HMC backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from src.action import (
    action_gradient,
    euclidean_lattice_action,
    local_delta_action,
)
from src.config import LatticeParams, PotentialParams, RunConfig
from src.utils import make_rng

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PIMCChain:
    """Results for one Markov chain.

    Attributes:
        samples: Stored configurations with shape `(n_samples, N)`.
        acceptance_rate: Total acceptance fraction over all proposals.
        final_proposal_width: Alias for the final sampler control value. For
            `method="metropolis_rb"` this is the tuned
            proposal width; for `method="hmc"` this is the tuned HMC step size.
        metadata: Run metadata for reproducibility and diagnostics.
    """

    samples: FloatArray
    acceptance_rate: float
    final_proposal_width: float
    metadata: dict[str, Any]

    @property
    def final_control_value(self) -> float:
        """Return the method-agnostic final sampler control value."""
        return self.final_proposal_width


def _vectorized_subset_update(
    path: FloatArray,
    indices: IntArray,
    proposal_width: float,
    potential: PotentialParams,
    lattice: LatticeParams,
    rng: np.random.Generator,
) -> int:
    """Apply one vectorized Metropolis sub-step for a parity subset."""
    proposals = path[indices] + rng.uniform(
        low=-proposal_width,
        high=proposal_width,
        size=indices.size,
    )

    delta_s = local_delta_action(
        path=path,
        proposed_values=proposals,
        indices=indices,
        potential=potential,
        lattice=lattice,
    )

    # Compare in log-space for robust acceptance handling.
    log_u = np.log(rng.random(indices.size))
    accepted = log_u < -delta_s

    if np.any(accepted):
        path[indices[accepted]] = proposals[accepted]

    return int(np.count_nonzero(accepted))


def _tune_proposal_width(
    current_width: float,
    acceptance: float,
    target_accept: float,
) -> float:
    """Adapt proposal width using a smooth bounded multiplicative controller."""
    error = acceptance - target_accept
    scale = float(np.exp(0.35 * error))
    scale = min(1.25, max(0.8, scale))
    return max(1e-12, current_width * scale)


def _hmc_full_path_update(
    path: FloatArray,
    *,
    step_size: float,
    n_leapfrog: int,
    mass: float,
    potential: PotentialParams,
    lattice: LatticeParams,
    rng: np.random.Generator,
) -> int:
    """Apply one full-path HMC proposal using leapfrog integration."""
    momentum = rng.normal(loc=0.0, scale=np.sqrt(mass), size=path.size)
    current_action = euclidean_lattice_action(
        path,
        potential=potential,
        lattice=lattice,
    )
    current_kinetic = 0.5 * float(np.sum(momentum**2)) / mass
    current_hamiltonian = current_action + current_kinetic

    proposal_path = path.copy()
    proposal_momentum = momentum.copy()

    grad = action_gradient(proposal_path, potential=potential, lattice=lattice)
    proposal_momentum -= 0.5 * step_size * grad

    for step in range(n_leapfrog):
        proposal_path += (step_size / mass) * proposal_momentum
        grad = action_gradient(proposal_path, potential=potential, lattice=lattice)
        if step < n_leapfrog - 1:
            proposal_momentum -= step_size * grad

    proposal_momentum -= 0.5 * step_size * grad
    proposal_momentum = -proposal_momentum

    proposal_action = euclidean_lattice_action(
        proposal_path,
        potential=potential,
        lattice=lattice,
    )
    proposal_kinetic = 0.5 * float(np.sum(proposal_momentum**2)) / mass
    proposal_hamiltonian = proposal_action + proposal_kinetic

    log_u = np.log(rng.random())
    if log_u < -(proposal_hamiltonian - current_hamiltonian):
        path[:] = proposal_path
        return 1
    return 0


def run_pimc(
    config: RunConfig,
    initial_path: ArrayLike | None = None,
    progress: bool = False,
) -> PIMCChain:
    """Run a deterministic PIMC chain using the configured sampling backend."""
    n_slices = config.lattice.n_slices
    method = config.sampler.method
    rng = make_rng(config.sampler.seed, set_hash_seed=True)

    if initial_path is None:
        path = np.zeros(n_slices, dtype=np.float64)
    else:
        path = np.asarray(initial_path, dtype=np.float64).copy()

    if path.ndim != 1:
        raise ValueError("initial_path must be one-dimensional")
    if path.size != n_slices:
        raise ValueError("initial_path length must match lattice.n_slices")

    if method == "metropolis_rb":
        if n_slices % 2 != 0:
            raise ValueError(
                "metropolis_rb requires an even n_slices for exact red-black "
                "updates with periodic boundaries; use method='hmc' or choose "
                "an even lattice size."
            )
        even_indices = np.arange(0, n_slices, 2, dtype=np.int64)
        odd_indices = np.arange(1, n_slices, 2, dtype=np.int64)
    else:
        even_indices = np.empty(0, dtype=np.int64)
        odd_indices = np.empty(0, dtype=np.int64)

    total_sweeps = config.total_sweeps
    n_therm = config.sampler.n_therm
    measure_every = config.sampler.measure_every
    if method == "metropolis_rb":
        control_value = float(config.sampler.proposal_width)
    else:
        control_value = float(config.sampler.hmc_step_size)

    n_samples = config.n_samples
    samples = np.empty((n_samples, n_slices), dtype=np.float64)
    sample_cursor = 0

    accepted_total = 0
    proposals_total = 0

    accepted_window = 0
    proposals_window = 0

    iterator: range | Any = range(total_sweeps)
    if progress:
        from tqdm.auto import trange

        iterator = trange(total_sweeps, desc="PIMC sweeps", leave=False)

    for sweep in iterator:
        proposals_sweep: int
        if method == "metropolis_rb":
            accepted_sweep = 0
            accepted_sweep += _vectorized_subset_update(
                path=path,
                indices=even_indices,
                proposal_width=control_value,
                potential=config.potential,
                lattice=config.lattice,
                rng=rng,
            )
            accepted_sweep += _vectorized_subset_update(
                path=path,
                indices=odd_indices,
                proposal_width=control_value,
                potential=config.potential,
                lattice=config.lattice,
                rng=rng,
            )
            proposals_sweep = n_slices
        else:
            accepted_sweep = _hmc_full_path_update(
                path=path,
                step_size=control_value,
                n_leapfrog=config.sampler.hmc_n_leapfrog,
                mass=config.sampler.hmc_mass,
                potential=config.potential,
                lattice=config.lattice,
                rng=rng,
            )
            proposals_sweep = 1

        accepted_total += accepted_sweep
        proposals_total += proposals_sweep

        # Adaptive proposal tuning is restricted to thermalization sweeps.
        if config.sampler.tune and sweep < n_therm:
            accepted_window += accepted_sweep
            proposals_window += proposals_sweep

            if (sweep + 1) % config.sampler.tune_interval == 0:
                recent_accept = accepted_window / float(proposals_window)
                control_value = _tune_proposal_width(
                    current_width=control_value,
                    acceptance=recent_accept,
                    target_accept=config.sampler.target_accept,
                )
                accepted_window = 0
                proposals_window = 0

        if sweep >= n_therm:
            meas_sweep = sweep - n_therm
            if meas_sweep % measure_every == 0:
                samples[sample_cursor, :] = path
                sample_cursor += 1

    if sample_cursor != n_samples:
        samples = samples[:sample_cursor]

    acceptance_rate = 0.0
    if proposals_total > 0:
        acceptance_rate = accepted_total / float(proposals_total)

    metadata: dict[str, Any] = {
        "run_name": config.run_name,
        "method": method,
        "beta": config.lattice.beta,
        "n_slices": config.lattice.n_slices,
        "a": config.lattice.a,
        "mass": config.potential.m,
        "omega": config.potential.omega,
        "lam": config.potential.lam,
        "n_therm": config.sampler.n_therm,
        "n_sweeps": config.sampler.n_sweeps,
        "measure_every": config.sampler.measure_every,
        "seed": config.sampler.seed,
        "tune": config.sampler.tune,
        "tune_interval": config.sampler.tune_interval,
        "target_accept": config.sampler.target_accept,
        "proposal_width": config.sampler.proposal_width,
        "hmc_step_size": config.sampler.hmc_step_size,
        "hmc_n_leapfrog": config.sampler.hmc_n_leapfrog,
        "hmc_mass": config.sampler.hmc_mass,
        "final_control_name": (
            "proposal_width" if method == "metropolis_rb" else "hmc_step_size"
        ),
        "final_control_value": float(control_value),
    }

    return PIMCChain(
        samples=samples,
        acceptance_rate=float(acceptance_rate),
        final_proposal_width=float(control_value),
        metadata=metadata,
    )
