"""Potential-energy utilities for the 1D quartic anharmonic oscillator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pimc_oscillator.config import PotentialParams

FloatArray = NDArray[np.float64]


def potential_energy(x: ArrayLike, params: PotentialParams) -> FloatArray:
    """Evaluate V(x) = 0.5*m*omega^2*x^2 + lam*x^4.

    Args:
        x: Scalar or array-like coordinates.
        params: Potential parameters (m, omega, lam).

    Returns:
        NumPy array with the same broadcasted shape as the input.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    harmonic = 0.5 * params.m * (params.omega**2) * x_arr**2
    anharmonic = params.lam * x_arr**4
    return harmonic + anharmonic


def potential_force(x: ArrayLike, params: PotentialParams) -> FloatArray:
    """Evaluate dV/dx for the quartic anharmonic potential."""
    x_arr = np.asarray(x, dtype=np.float64)
    return params.m * (params.omega**2) * x_arr + 4.0 * params.lam * x_arr**3


@dataclass(frozen=True, slots=True)
class AnharmonicPotential:
    """Object-oriented wrapper around quartic potential parameters."""

    params: PotentialParams

    def evaluate(self, x: ArrayLike) -> FloatArray:
        """Evaluate the potential at one or more positions."""
        return potential_energy(x, self.params)

    def force(self, x: ArrayLike) -> FloatArray:
        """Evaluate dV/dx at one or more positions."""
        return potential_force(x, self.params)
