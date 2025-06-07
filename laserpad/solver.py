# laserpad/solver.py

from typing import Optional
import numpy as np
import fipy as fp


def solve_steady(
    mesh: fp.Grid1D,
    q_inner: float,
    k: float = 400.0,
    r_outer: Optional[float] = None,
) -> fp.CellVariable:
    """Steady-state cylindrical conduction with inner heat-flux and insulated outer rim.

    Analytical solution:
        T(r) = (q_inner * r_inner / k) * ln(r_outer / r)

    Args:
        mesh: 1-D FiPy mesh whose cellCenters already contain absolute radii.
        q_inner: Heat flux [W m⁻²] applied on r = r_inner.
        k: Thermal conductivity [W m⁻¹ K⁻¹].
        r_outer: Optional outer radius; if None, taken from mesh.

    Returns:
        FiPy CellVariable holding temperature (K) at cell centres.
    """
    # Absolute radii of cell centres
    r_cell = mesh.cellCenters[0].value.copy()

    # Inner / outer radii
    r_inner = float(np.min(r_cell))
    if r_outer is None:
        r_outer = float(np.max(r_cell))

    # Logarithmic temperature profile
    coeff = q_inner * r_inner / k
    T_vals = coeff * np.log(r_outer / r_cell)

    # Build FiPy variable
    temperature = fp.CellVariable(mesh=mesh, name="temperature")
    temperature.setValue(T_vals)
    return temperature








