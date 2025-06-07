# laserpad/solver.py

from typing import Optional

import numpy as np
import fipy as fp


def solve_steady(
    mesh: fp.Grid1D, q_inner: float, k: float = 400.0, r_outer: Optional[float] = None
) -> fp.CellVariable:
    """Compute steady-state T(r) in a cylindrical annulus with inner heat-flux and insulated outer rim."""
    import numpy as np

    # Extract absolute radii of cell centers
    r_cell = mesh.cellCenters[0].value.copy()

    # Determine r_inner and r_outer
    r_inner = float(np.min(r_cell))
    if r_outer is None:
        r_outer = float(np.max(r_cell))

    # Analytical solution in cylindrical coords with:
    #   1/r d/dr (r k dT/dr) = 0,
    #   -k dT/dr|_{r_inner} = q_inner,
    #   dT/dr|_{r_outer} = 0  (insulated)
    #
    # Solution: T(r) = (q_inner * r_inner / k) * ln(r_outer / r)
    coeff = q_inner * r_inner / k
    T_vals = coeff * np.log(r_outer / r_cell)

    # Populate FiPy variable
    temperature = fp.CellVariable(mesh=mesh, name="temperature")
    temperature.setValue(T_vals)
    return temperature

