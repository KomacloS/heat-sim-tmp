# laserpad/solver.py
import fipy as fp
import numpy as np
from typing import Optional

def solve_steady(mesh: fp.Grid1D,
                 q_inner: float,
                 k: float = 400.0,
                 r_outer: Optional[float] = None,
                 h: float = 1_000.0,
                 T_inf: float = 0.0) -> fp.CellVariable:
    """Steady 1-D conduction in a ring; all units SI."""
    r = mesh.cellCenters[0]          # absolute radii [m]
    dr = float(mesh.dx)
    if r_outer is None:
        r_outer = float(r.max() + dr/2)

    T = fp.CellVariable(mesh=mesh, value=T_inf, name="temperature")

    # inner heat-flux  (−k∂T/∂r = q_inner)
    T.faceGrad.constrain((-q_inner / k,), where=mesh.facesLeft)

    # mask that activates the Robin term only in the outermost cell
    beta = np.zeros(mesh.numberOfCells)
    beta[-1] = 1.0
    beta = fp.CellVariable(mesh=mesh, value=beta)

    # **Cylindrical operator**  — note the extra factor r everywhere
    eq = (
        fp.DiffusionTerm(coeff=k * r, var=T)          # ∂/∂r( r k ∂T/∂r )
        + fp.ImplicitSourceTerm(coeff=beta * h * r,   #   + β h r T
                                var=T)
        - beta * h * r * T_inf                        # RHS: β h r T∞
    )

    eq.solve(var=T)
    return T
