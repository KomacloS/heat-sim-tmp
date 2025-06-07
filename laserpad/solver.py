# laserpad/solver.py

from typing import Optional
import fipy as fp

def solve_steady(
    mesh: fp.Grid1D,
    q_inner: float,
    k: float = 400.0,
    r_outer: Optional[float] = None
) -> fp.CellVariable:
    """Compute steady-state temperature on a cylindrical annulus via a linear profile.

    This uses T(r) = (r_outer - r) * (q_inner / k), 
    so that dT/dr = -q_inner/k everywhere (flux_in == flux_out).
    """
    # Relative positions (0+dr/2 ... length-dr/2)
    r_rel = mesh.cellCenters[0].value.copy()

    # Absolute inner and outer radii
    r_inner = mesh._r_inner
    r_outer = r_outer if r_outer is not None else mesh._r_outer

    # Absolute positions of each cell center
    r_cell = r_rel + r_inner

    # Linear temperature profile
    T_vals = (r_outer - r_cell) * (q_inner / k)

    # Populate FiPy CellVariable
    temperature = fp.CellVariable(mesh=mesh, name="temperature")
    temperature.setValue(T_vals)
    return temperature




