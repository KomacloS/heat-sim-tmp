# laserpad/solver.py
import fipy as fp
from typing import Optional

def solve_steady(
    mesh: fp.Grid1D,
    q_inner: float,
    k: float = 400.0,
    r_outer: Optional[float] = None,
    h: float = 1_000.0,
    T_inf: float = 0.0,
) -> fp.CellVariable:
    """Steady 1-D conduction with a Robin BC on the outer rim (SI units)."""

    # radii bookkeeping ---------------------------------------------------
    dr      = float(mesh.dx)
    r_cells = mesh.cellCenters[0].value
    r_inner = float(r_cells.min() - dr / 2.0)
    if r_outer is None:
        r_outer = float(r_cells.max() + dr / 2.0)

    # unknown --------------------------------------------------------------
    T = fp.CellVariable(mesh=mesh, name="temperature", value=T_inf)

    # inner Neumann:  -k dT/dr = q_inner  ---------------------------------
    T.faceGrad.constrain((-q_inner / k,), where=mesh.facesLeft)

    # ---- outer Robin via last-cell term ---------------------------------
    beta = fp.CellVariable(mesh=mesh, value=0.0)
    beta[-1] = 1.0             # 1 in last cell, 0 elsewhere

    ε = 1.0e-12                # tiny reaction in *every* cell

    eq = (
        fp.DiffusionTerm(coeff=k, var=T)
        + fp.ImplicitSourceTerm(coeff=beta * h / k, var=T)   # Robin
        + fp.ImplicitSourceTerm(coeff=ε,           var=T)    # stabiliser
    )

    eq.solve(var=T)
    return T
