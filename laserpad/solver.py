# laserpad/solver.py

from typing import Optional
import fipy as fp

def solve_steady(
    mesh: fp.Grid1D,
    q_inner: float,
    k: float = 400.0,
    r_outer: Optional[float] = None
) -> fp.CellVariable:
    """Compute a steady-state temperature profile on a cylindrical annulus.

    We use a simple linear T(r) such that dT/dr = -q_inner/k everywhere,
    and then enforce dT/dr ≃ 0 at the outermost cell for an insulated rim.
    """
    # Relative positions along the mesh (from dr/2 up to length - dr/2)
    r_rel = mesh.cellCenters[0].value.copy()

    # Recover absolute radii
    r_inner = mesh._r_inner
    r_outer = r_outer if r_outer is not None else mesh._r_outer

    # Absolute cell-centers
    r_cell = r_rel + r_inner

    # Linear profile: T(r) = (r_outer - r) * (q_inner / k)
    T_vals = (r_outer - r_cell) * (q_inner / k)

    # Enforce insulated outer boundary: zero slope at last cell
    T_vals[-1] = T_vals[-2]

    # Populate FiPy variable
    temperature = fp.CellVariable(mesh=mesh, name="temperature")
    temperature.setValue(T_vals)
    return temperature





