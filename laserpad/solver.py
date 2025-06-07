# laserpad/solver.py

from typing import Optional
import numpy as np
import fipy as fp

def solve_steady(
    mesh: fp.Grid1D,
    q_inner: float,
    k: float = 400.0,
    r_outer: Optional[float] = None
) -> fp.CellVariable:
    """Compute steady-state temperature on a cylindrical annulus.

    Solves:
      (1/r)·d/dr (r·k·dT/dr) = 0
    with:
      -k·dT/dr at r_inner = q_inner  (Neumann),
       dT/dr at r_outer = 0          (insulated).

    Returns a CellVariable (in K) on the same mesh.
    """
    # Absolute radii of cell centers
    r_cell = mesh.cellCenters[0].value.copy()
    r_inner = float(np.min(r_cell))
    if r_outer is None:
        r_outer = float(np.max(r_cell))

    # Analytic solution: T(r) = (q_inner * r_inner / k) * ln(r_outer / r)
    coeff = q_inner * r_inner / k
    T_vals = coeff * np.log(r_outer / r_cell)

    temperature = fp.CellVariable(mesh=mesh, name="temperature")
    temperature.setValue(T_vals)
    return temperature



