"""Temperature solver for the radial conduction problem."""

from __future__ import annotations

import numpy as np

from .geometry import Mesh


def solve(
    mesh: Mesh,
    q_inner: float,
    k: float = 400.0,
    h: float = 1_000.0,
    T_inf: float = 0.0,
) -> np.ndarray:
    """Compute the steady-state temperature profile.

    The analytic solution of the radial conduction equation with an imposed
    inner heat flux and an outer convective boundary condition is used. The
    function returns the temperature at the mesh cell centres.
    """

    r_inner = mesh.r_inner
    r_outer = mesh.r_outer
    r = mesh.r

    coeff = q_inner * r_inner / k
    B = T_inf + coeff * np.log(r_outer) + (q_inner * r_inner) / (h * r_outer)
    return B - coeff * np.log(r)


# Backwards compatibility
def solve_steady(
    mesh: Mesh,
    q_inner: float,
    k: float = 400.0,
    r_outer: float | None = None,
    h: float = 1_000.0,
    T_inf: float = 0.0,
    r_inner: float | None = None,
) -> np.ndarray:
    """Compatibility wrapper retaining the old API."""

    return solve(mesh, q_inner=q_inner, k=k, h=h, T_inf=T_inf)
