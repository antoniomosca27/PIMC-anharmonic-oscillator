"""Euclidean lattice action and local action-difference utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pimc_oscillator.config import LatticeParams, PotentialParams
from pimc_oscillator.potential import potential_energy, potential_force

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def euclidean_lattice_action(
    path: ArrayLike,
    potential: PotentialParams,
    lattice: LatticeParams,
) -> float:
    """Compute the discretized Euclidean action with periodic boundaries.

    The action convention is:
        S = sum_i [ (m/(2a))*(x_{i+1}-x_i)^2 + a*V(x_i) ]
    where `a = beta / N` and `x_N = x_0`.
    """
    x = np.asarray(path, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("path must be one-dimensional")
    if x.size != lattice.n_slices:
        raise ValueError("path length must match lattice.n_slices")

    a = lattice.a
    forward_diff = np.roll(x, -1) - x
    kinetic = (potential.m / (2.0 * a)) * forward_diff**2
    potential_term = a * potential_energy(x, potential)
    return float(np.sum(kinetic + potential_term))


def local_delta_action(
    path: FloatArray,
    proposed_values: ArrayLike,
    indices: ArrayLike,
    potential: PotentialParams,
    lattice: LatticeParams,
) -> FloatArray:
    """Compute local action differences for a vectorized site subset.

    For each selected site i, this function returns:
        ΔS_i = [m/(2a)] * (
            (x_i' - x_{i-1})^2 + (x_{i+1} - x_i')^2
            - (x_i - x_{i-1})^2 - (x_{i+1} - x_i)^2
        ) + a * (V(x_i') - V(x_i)).

    This expression is exact for single-site updates under periodic
    boundary conditions and is suitable for red-black even/odd updates.
    """
    if path.ndim != 1:
        raise ValueError("path must be one-dimensional")
    if path.size != lattice.n_slices:
        raise ValueError("path length must match lattice.n_slices")

    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("indices must be one-dimensional")

    x_new = np.asarray(proposed_values, dtype=np.float64)
    if x_new.shape != idx.shape:
        raise ValueError("proposed_values must have the same shape as indices")

    n = lattice.n_slices
    a = lattice.a
    pref = potential.m / (2.0 * a)

    x_old = path[idx]
    x_prev = path[(idx - 1) % n]
    x_next = path[(idx + 1) % n]

    old_kinetic = (x_old - x_prev) ** 2 + (x_next - x_old) ** 2
    new_kinetic = (x_new - x_prev) ** 2 + (x_next - x_new) ** 2
    delta_kinetic = pref * (new_kinetic - old_kinetic)

    delta_potential = a * (
        potential_energy(x_new, potential) - potential_energy(x_old, potential)
    )
    return delta_kinetic + delta_potential


def action_gradient(
    path: ArrayLike,
    potential: PotentialParams,
    lattice: LatticeParams,
) -> FloatArray:
    """Compute dS/dx for the full path with periodic boundaries.

    For
        S = sum_i [ (m/(2a)) * (x_{i+1} - x_i)^2 + a * V(x_i) ],
    the gradient is
        dS/dx_i = (m/a) * (2*x_i - x_{i-1} - x_{i+1}) + a * dV/dx_i.
    """
    x = np.asarray(path, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("path must be one-dimensional")
    if x.size != lattice.n_slices:
        raise ValueError("path length must match lattice.n_slices")

    a = lattice.a
    kinetic_grad = (potential.m / a) * (2.0 * x - np.roll(x, 1) - np.roll(x, -1))
    potential_grad = a * potential_force(x, potential)
    return kinetic_grad + potential_grad
