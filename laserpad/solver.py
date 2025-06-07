import fipy as fp
import numpy as np
from typing import Optional

def solve_steady(
    mesh: fp.Grid1D,
    q_inner: float,
    k: float = 400.0,
    r_outer: Optional[float] = None,
    h: float = 1_000.0,
    T_inf: float = 0.0,
) -> fp.CellVariable:
    """
    Analytical steady radial conduction.
    """
    dr      = float(mesh.dx)
    r_cells = mesh.cellCenters[0].value
    r_inner = float(r_cells.min() - dr/2)
    if r_outer is None:
        r_outer = float(r_cells.max() + dr/2)

    # Analytic solution
    T_vals = (q_inner * r_inner / k) * np.log(r_outer / r_cells) \
           + (q_inner * r_inner) / (h * r_outer) \
           + T_inf

    return fp.CellVariable(mesh=mesh, name="temperature", value=T_vals)
