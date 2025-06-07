# laserpad/solver.py
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
    Cylindrical, steady-state conduction solved with FiPy FVM.

    Inner BC (r = r_inner):
        −k ∂T/∂r  =  q_inner          (Neumann)

    Outer BC (r = r_outer):
        −k ∂T/∂r  =  h (T − T_inf)    (Robin)

    All inputs are **SI**.  The mesh must already hold absolute radii
    (``build_mesh`` does the origin shift).
    """
    # --- radii ----------------------------------------------------------
    dr       = float(mesh.dx)
    r_cell   = mesh.cellCenters[0].value
    r_inner  = float(r_cell.min() - dr / 2.0)
    r_outer  = float(r_cell.max() + dr / 2.0) if r_outer is None else r_outer

    # --- unknown --------------------------------------------------------
    T = fp.CellVariable(mesh=mesh, name="temperature", value=T_inf)

    # ---- inner Neumann:  ∂T/∂r = −q/k  ---------------------------------
    T.faceGrad.constrain((-q_inner / k,), where=mesh.facesLeft)

    # ---- outer Robin via last-cell source terms ------------------------
    beta = fp.CellVariable(mesh=mesh, value=0.0)   # 1 only in the last cell
    beta[-1] = 1.0

    eq = (
        fp.DiffusionTerm(coeff=k)                         # 1/r d/dr(r k dT/dr)
        + fp.ImplicitSourceTerm(coeff=beta * h / k)       #  h · T
        + (beta * h * T_inf / k)                          # −h · T_inf
    )

    eq.solve(var=T, cacheMatrix=False)   # < 1 ms for 100 cells

    return T
