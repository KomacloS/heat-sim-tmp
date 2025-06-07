# laserpad/solver.py

from typing import Optional
import fipy as fp
import numpy as np

def solve_steady(
    mesh: fp.Grid1D, q_inner: float, k: float = 400.0, r_outer: Optional[float] = None
) -> fp.CellVariable:
    """Compute steady-state temperature on a cylindrical annulus.

    Solves
        (1/r) d/dr (r k dT/dr) = 0
    with
        -k dT/dr at r = r_inner equals q_inner  (inner Neumann),
         dT/dr at r = r_outer equals 0           (outer insulation).

    Returns a CellVariable (in K) on the same mesh.
    """
    # Get relative cell-centers (0+dr/2 ... length-dr/2)
    r_rel = mesh.cellCenters[0].value.copy()

    # Recover the true radii
    r_inner = mesh._r_inner
    if r_outer is None:
        r_outer = mesh._r_outer

    # Absolute cell-center locations
    r_cell = r_rel + r_inner

    # Analytical solution in cylindrical coords:
    #   T(r) = (q_inner * r_inner / k) * ln(r_outer / r)
    coeff = q_inner * r_inner / k
    T_vals = coeff * np.log(r_outer / r_cell)

    # Populate FiPy variable
    temperature = fp.CellVariable(mesh=mesh, name="temperature")
    temperature.setValue(T_vals)

    return temperature


