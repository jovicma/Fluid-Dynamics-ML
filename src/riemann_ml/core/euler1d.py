"""Core utilities for the one-dimensional Euler equations.

The routines implemented here follow the notation from:

* E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
  Springer, 3rd edition, 2009.

All variables refer to the conservative vector ``U = [ρ, ρu, E]`` and the
primitive state ``W = [ρ, u, p]`` for an ideal gas with ratio of specific heats
``γ`` (``gamma``). Robustness is ensured by clamping density and pressure to a
small positive number when performing transformations.
"""

from __future__ import annotations

import math
from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field

EPSILON: float = 1e-12


class StatePrim(BaseModel):
    """Primitive state ``W = [ρ, u, p]`` for the Euler equations.

    Attributes
    ----------
    density:
        Fluid density ``ρ`` (kg/m³), enforced to be positive.
    velocity:
        Fluid velocity ``u`` (m/s).
    pressure:
        Thermodynamic pressure ``p`` (Pa), enforced to be positive.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    density: float = Field(..., gt=0.0)
    velocity: float
    pressure: float = Field(..., gt=0.0)


class StateCons(BaseModel):
    """Conservative state ``U = [ρ, ρu, E]`` for the Euler equations.

    Attributes
    ----------
    density:
        Mass density ``ρ`` (kg/m³), enforced to be positive.
    momentum:
        Momentum density ``ρu`` (kg/(m²·s)).
    energy:
        Total energy density ``E`` (J/m³), enforced to be positive.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    density: float = Field(..., gt=0.0)
    momentum: float
    energy: float = Field(..., gt=0.0)


def _clamp_positive(value: float, *, name: str, eps: float = EPSILON) -> float:
    """Clamp the scalar ``value`` to be strictly greater than ``eps``."""
    if value <= eps:
        return eps
    return value


def prim_to_cons(prim: StatePrim, gamma: float = 1.4) -> StateCons:
    r"""Convert primitive variables to conservative variables.

    Parameters
    ----------
    prim:
        Primitive state ``[ρ, u, p]``.
    gamma:
        Ratio of specific heats (``γ``), defaulting to 1.4 for air.

    Returns
    -------
    StateCons
        Conservative state ``[ρ, ρu, E]`` where the total energy density is
        given by :math:`E = \\frac{p}{γ-1} + \\tfrac{1}{2} ρ u^2`.

    References
    ----------
    Toro (2009), Eq. (2.7).
    """
    rho = _clamp_positive(prim.density, name="density")
    pressure = _clamp_positive(prim.pressure, name="pressure")
    velocity = prim.velocity

    momentum = rho * velocity
    kinetic = 0.5 * rho * velocity**2
    internal = pressure / (gamma - 1.0)
    energy = _clamp_positive(internal + kinetic, name="energy")

    return StateCons(density=rho, momentum=momentum, energy=energy)


def pressure_from_cons(cons: StateCons, gamma: float = 1.4) -> float:
    r"""Recover the thermodynamic pressure from the conservative state.

    The relation used is :math:`p = (γ-1) (E - \\tfrac{1}{2} ρ u^2)`. Numerical
    safeguards ensure that negative pressures are avoided by clamping the
    internal energy to a tiny positive value.
    """
    rho = _clamp_positive(cons.density, name="density")
    energy = _clamp_positive(cons.energy, name="energy")
    velocity = cons.momentum / rho

    kinetic = 0.5 * rho * velocity**2
    internal_energy = max(energy - kinetic, EPSILON / (gamma - 1.0))
    pressure = (gamma - 1.0) * internal_energy
    return _clamp_positive(pressure, name="pressure")


def cons_to_prim(cons: StateCons, gamma: float = 1.4) -> StatePrim:
    """Convert conservative variables to primitive variables."""
    rho = _clamp_positive(cons.density, name="density")
    velocity = cons.momentum / rho
    pressure = pressure_from_cons(cons, gamma=gamma)
    return StatePrim(density=rho, velocity=velocity, pressure=pressure)


def sound_speed(state: StatePrim | StateCons, gamma: float = 1.4) -> float:
    """Compute the acoustic speed ``c = sqrt(γ p / ρ)``.

    Parameters
    ----------
    state:
        Either a primitive or conservative state.
    gamma:
        Ratio of specific heats.
    """
    if isinstance(state, StateCons):
        prim = cons_to_prim(state, gamma=gamma)
    else:
        prim = state
    rho = _clamp_positive(prim.density, name="density")
    pressure = _clamp_positive(prim.pressure, name="pressure")
    return math.sqrt(gamma * pressure / rho)


def flux_vector(cons: StateCons, gamma: float = 1.4) -> Tuple[float, float, float]:
    r"""Compute the Euler flux vector ``F(U)``.

    The physical flux is

    .. math::
        F(U) = \begin{bmatrix}
            ρu \\
            ρu^2 + p \\
            u(E + p)
        \end{bmatrix}.

    Parameters
    ----------
    cons:
        Conservative state ``[ρ, ρu, E]``.
    gamma:
        Ratio of specific heats used to recover the pressure.
    """
    rho = _clamp_positive(cons.density, name="density")
    momentum = cons.momentum
    velocity = momentum / rho
    pressure = pressure_from_cons(cons, gamma=gamma)
    energy = _clamp_positive(cons.energy, name="energy")

    mass_flux = momentum
    momentum_flux = momentum * velocity + pressure
    energy_flux = velocity * (energy + pressure)
    return mass_flux, momentum_flux, energy_flux
