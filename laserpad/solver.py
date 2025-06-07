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
    Steady-state 1-D cylindrical conduction solved with FiPy FVM.

    BCs
    ----
    • Inner face (r = r_inner) :  −k ∂T/∂r = q_inner          (Neumann)
    • Outer face (r = r_outer) :  −k ∂T/∂r = h (T − T_inf)    (Robin)

    All arguments are **SI units**.  The mesh returned by `build_mesh`
    already stores absolute radii, so no conversion is needed here.
    """
    # ---------------------------------------------------------------------
    # Radii bookkeeping
    dr       = float(mesh.dx)
    r_cells  = mesh.cellCenters[0].value
    r_inner  = float(r_cells.min() - dr / 2.0)
    if r_outer is None:
        r_outer = float(r_cells.max() + dr / 2.0)

    # ---------------------------------------------------------------------
    # Unknown field
    T = fp.CellVariable(mesh=mesh, name="temperature", value=T_inf)

    # ---------------------------------------------------------------------
    # Inner Neumann: ∂T/∂r = −q/k  (imposed on the inner face)
    T.faceGrad.constrain((-q_inner / k,), where=mesh.facesLeft)

    # ---------------------------------------------------------------------
    # Outer Robin implemented as last-cell source terms
    beta = fp.CellVariable(mesh=mesh, value=0.0)   # 1 only in last cell
    beta[-1] = 1.0

    # Governing equation:
    # --- build PDE:  (1/r)∂/∂r(r k ∂T/∂r)  +  β h/k · T  =  β h/k · T_inf  ----------
    eq = (
        fp.DiffusionTerm(coeff=k, var=T)                 # conduction
        + fp.ImplicitSourceTerm(coeff=beta * h / k, var=T)   # Robin     (h·T)
        # + explicit term is zero for T_inf = 0, so omit it
    )



    # ---------------------------------------------------------------------
    eq.solve(var=T)
    return T

