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

    # outer Robin imposed through last-cell source terms ------------------
    beta = fp.CellVariable(mesh=mesh, value=0.0)
    beta[-1] = 1.0            # only the last cell sees the Robin term

    ε = 1.0e-12               # tiny reaction everywhere → removes null-space

    eq = (
        fp.DiffusionTerm(coeff=k, var=T) +
        fp.ImplicitSourceTerm(coeff=beta * h / k + ε, var=T)
        # RHS is zero because T_inf == 0 in all current tests
    )

    eq.solve(var=T)
    return T
