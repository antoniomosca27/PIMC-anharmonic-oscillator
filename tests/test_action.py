"""Tests for Euclidean action and local action-difference calculations."""

from __future__ import annotations

import numpy as np

from src import LatticeParams, PotentialParams
from src.action import (
    action_gradient,
    euclidean_lattice_action,
    local_delta_action,
)


def test_constant_path_action_matches_beta_times_potential() -> None:
    """For a constant path, verify kinetic contribution is exactly zero."""
    lattice = LatticeParams(beta=3.7, n_slices=24)
    potential = PotentialParams(m=1.4, omega=1.2, lam=0.6)

    constant_value = 0.42
    path = np.full(lattice.n_slices, constant_value, dtype=np.float64)

    action = euclidean_lattice_action(path, potential=potential, lattice=lattice)

    v_const = 0.5 * potential.m * (potential.omega**2) * constant_value**2
    v_const += potential.lam * constant_value**4
    expected = lattice.beta * v_const

    assert np.isclose(action, expected, rtol=0.0, atol=1e-12)


def test_local_delta_action_matches_full_action_difference(
    rng: np.random.Generator,
    lattice_params: LatticeParams,
    potential_params: PotentialParams,
) -> None:
    """A one-site local delta action must match the full action difference."""
    path = rng.normal(loc=0.0, scale=0.6, size=lattice_params.n_slices)

    site = 7
    proposal = path[site] + 0.137

    delta_local = local_delta_action(
        path=path,
        proposed_values=np.array([proposal], dtype=np.float64),
        indices=np.array([site], dtype=np.int64),
        potential=potential_params,
        lattice=lattice_params,
    )[0]

    updated = path.copy()
    updated[site] = proposal

    delta_full = euclidean_lattice_action(
        updated,
        potential=potential_params,
        lattice=lattice_params,
    ) - euclidean_lattice_action(
        path,
        potential=potential_params,
        lattice=lattice_params,
    )

    assert np.isclose(delta_local, delta_full, rtol=0.0, atol=1e-11)


def test_action_gradient_matches_finite_difference() -> None:
    """Full action gradient should agree with central finite differences."""
    lattice = LatticeParams(beta=2.3, n_slices=8)
    potential = PotentialParams(m=1.1, omega=0.9, lam=0.25)
    rng = np.random.Generator(np.random.PCG64(101))
    path = rng.normal(loc=0.0, scale=0.4, size=lattice.n_slices)

    grad = action_gradient(path, potential=potential, lattice=lattice)

    eps = 1e-6
    grad_fd = np.empty_like(path)
    for i in range(path.size):
        plus = path.copy()
        minus = path.copy()
        plus[i] += eps
        minus[i] -= eps
        s_plus = euclidean_lattice_action(plus, potential=potential, lattice=lattice)
        s_minus = euclidean_lattice_action(minus, potential=potential, lattice=lattice)
        grad_fd[i] = (s_plus - s_minus) / (2.0 * eps)

    assert np.allclose(grad, grad_fd, rtol=1e-5, atol=1e-6)
