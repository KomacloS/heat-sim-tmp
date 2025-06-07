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

    # ---- outer Robin on last cell ----------------------------------------
    last = mesh.cellCenters[-1]           # mask for final cell
    beta = fp.CellVariable(mesh=mesh, value=0.0)
    beta[-1] = 1.0                        # 1 in last cell, 0 elsewhere

    eq = (
        fp.DiffusionTerm(coeff=k)                 # 1/r ∂/∂r(r k ∂T/∂r)
        + fp.ImplicitSourceTerm(coeff=beta * h / k, var=T)   #   h · T
        + (beta * h * T_inf / k)                  # – h · T_inf    (RHS)
        + fp.ImplicitSourceTerm(coeff=1e-12, var=T)          # ε·T anchor
    )

    eq.solve(var=T)     # < 2 ms for 100 cells
    return T
