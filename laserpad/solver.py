from typing import Optional

import fipy as fp
import numpy as np


def solve_steady(
    mesh: fp.Grid1D,
    q_inner: float,
    k: float = 400.0,
    r_outer: Optional[float] = None,
    h: float = 1_000.0,
    T_inf: float = 0.0,
) -> fp.CellVariable:
    """Steady-state cylindrical conduction solved with FiPy FVM.

    BCs
    ----
    • Inner rim :  −k ∂T/∂r = q_inner          (Neumann)
    • Outer rim :  −k ∂T/∂r = h (T − T_inf)    (Robin)
    """
    # ---- geometry ----
    dr = mesh.dx
    r_cell = mesh.cellCenters[0].value
    r_inner = float(r_cell.min())
    if r_outer is None:
        r_outer = float(r_cell.max() + dr / 2)

    # ---- variable ----
    T = fp.CellVariable(mesh=mesh, name="temperature", value=T_inf)

    # ---- inner Neumann ----
    T.faceGrad.constrain((-q_inner / k,), where=mesh.facesLeft)

    # ---- diffusion term (axisymmetric in 1-D) ----
    eq = fp.DiffusionTerm(coeff=k)

    # ---- outer Robin ----
    beta = fp.CellVariable(mesh=mesh, value=0.0)
    beta[-1] = 1.0

    eq += fp.ImplicitSourceTerm(coeff=(h / k) * beta)      # h · T
    eq += (h * T_inf / k) * beta                           # −h · T_inf  (constant)

    # ---- solve ----
    eq.solve(var=T)
    return T
