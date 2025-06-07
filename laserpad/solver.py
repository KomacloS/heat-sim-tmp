# laserpad/solver.py

from typing import Optional
import fipy as fp
import numpy as np

def solve_steady(
    mesh: fp.Grid1D,
    q_inner: float,
    k: float = 400.0,
    r_outer: Optional[float] = None
) -> fp.CellVariable:
    """Compute steady-state T(r) on a cylindrical annulus with insulated outer rim.

    Uses a quadratic profile:
        T(r) = A * (r_outer - r)^2,
    chosen so that
        -k·dT/dr|_{r_inner} = q_inner
    and
        dT/dr|_{r_outer} = 0.

    Args:
        mesh: 1D FiPy mesh with mesh._r_inner and mesh._r_outer set.
        q_inner: Heat flux at the inner boundary.
        k: Thermal conductivity.
        r_outer: Optional override for outer radius.

    Returns:
        CellVariable of temperatures.
    """
    # Get relative cell centers (0+dr/2 ... length−dr/2)
    r_rel = mesh.cellCenters[0].value.copy()

    # Recover absolute radii
    r_inner = mesh._r_inner
    if r_outer is None:
        r_outer = mesh._r_outer

    # Absolute positions
    r_cell = r_rel + r_inner

    # Compute coefficient A so that -k·dT/dr at r_inner = q_inner:
    #   dT/dr = -2·A·(r_outer - r), so at r_inner:
    #     -k * [ -2A(r_outer - r_inner) ] = q_inner
    #   ⇒ 2 A k (r_outer - r_inner) = q_inner
    A = q_inner / (2.0 * k * (r_outer - r_inner))

    # Quadratic temperature profile
    T_vals = A * (r_outer - r_cell) ** 2

    # Build FiPy variable
    temperature = fp.CellVariable(mesh=mesh, name="temperature")
    temperature.setValue(T_vals)
    return temperature





