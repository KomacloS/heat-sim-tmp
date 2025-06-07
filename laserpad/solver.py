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

    r = mesh.cellCenters[0]          # absolute radii [m]
    dr = float(mesh.dx)
    if r_outer is None:              # outer face radius
        r_outer = float(r.max() + dr/2)

    T = fp.CellVariable(mesh=mesh, value=T_inf, name="temperature")

    # --- inner Neumann (heat-flux) -----------------------------------------
    T.faceGrad.constrain((-q_inner/k,), where=mesh.facesLeft)

    # --- outer Robin encoded as a reaction term in the last cell -----------
    beta = np.zeros(mesh.numberOfCells, dtype=float)
    beta[-1] = 1.0                   # only outermost cell gets the Robin term
    beta = fp.CellVariable(mesh=mesh, value=beta)

    # Governing equation in divergence form (note the extra 'r')
    eq = (
        fp.DiffusionTerm(coeff=k * r, var=T)                       # ∂/∂r(r k ∂T/∂r)
        + fp.ImplicitSourceTerm(coeff=beta * h * r, var=T)         #   + β h r T
        - h * T_inf * r * beta                                     # RHS  β h r T∞
    )

    eq.solve(var=T)
    return T
