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
    # Extract radial positions of cell centers:
    r_cell = mesh.cellCenters[0].value.copy()

    # Determine r_inner and r_outer properly:
    dr = mesh.dx
    # Because we chose origin = r_inner - dr/2, the smallest center is r_inner:
    r_inner = float(np.min(r_cell))
    if r_outer is None:
        r_outer = float(r_inner + mesh.length)

    # C1 from BC at r_inner: -k * (C1/r_inner) = q_inner  --> C1 = -q_inner * r_inner / k
    C1 = -q_inner * r_inner / k
    # Impose Dirichlet at r_outer: T(r_outer) = 0 --> 0 = C1 * ln(r_outer) + C2
    C2 = -C1 * np.log(r_outer)

    # Analytic solution: T(r) = C1 * ln(r) + C2
    T_vals = C1 * np.log(r_cell) + C2

    # Build a FiPy CellVariable and assign these values:
    temperature = fp.CellVariable(mesh=mesh, name="temperature")
    temperature.setValue(T_vals)

    return temperature
