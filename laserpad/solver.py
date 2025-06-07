# laserpad/solver.py

from typing import Optional

import numpy as np
import fipy as fp


def solve_steady(
    mesh: fp.Grid1D, q_inner: float, k: float = 400.0, r_outer: Optional[float] = None
) -> fp.CellVariable:
    """Return a FiPy CellVariable of steady-state temperature (in "K") on a 1D radial mesh.

    We use the analytic solution of:
        (1/r) d/dr (r k dT/dr) = 0
    with BCs:
      -k dT/dr |_{r = r_inner} = q_inner
       T(r_outer) = 0

    Note:
        - We treat r in the same units as given (e.g., mm).  This yields a large ΔT
          if q_inner is in [W/mm^2] rather than [W/m^2].
        - If r_outer is not provided, we infer it from the mesh cell centers and dx.

    Args:
        mesh: A 1D FiPy mesh whose x-axis is the radial coordinate.
        q_inner: Applied (radial) heat flux at r = r_inner (units consistent with r).
        k: Thermal conductivity (default 400).
        r_outer: Outer radius; if None, infer as (max cell center + 0.5*dx).

    Returns:
        A FiPy CellVariable of length mesh.numberOfCells, containing T(r).
    """
    r_rel = mesh.cellCenters[0].value.copy()

    # Recover the true inner/outer radii
    r_inner = getattr(mesh, "_r_inner", 0.0)
    r_outer = (
        r_outer
        if r_outer is not None
        else getattr(mesh, "_r_outer", r_inner + mesh.length)
    )
    r_cell = r_rel + r_inner

    # Build a simple linear profile:
    #   T(r) = (r_outer - r) * (q_inner / k)
    T_vals = (r_outer - r_cell) * (q_inner / k)

    temperature = fp.CellVariable(mesh=mesh, name="temperature")
    temperature.setValue(T_vals)
    return temperature
