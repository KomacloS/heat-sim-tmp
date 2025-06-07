import math
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
    """Steady-state cylindrical conduction (FiPy FVM).

    BCs
    ----
    • Inner rim :  −k ∂T/∂r = q_inner      (Neumann)  
    • Outer rim :  −k ∂T/∂r = h (T − T_inf)  (Robin)

    Parameters
    ----------
    mesh   : 1-D `Grid1D` (cell centers = radii, m)
    q_inner: heat-flux at inner rim [W m⁻²], positive into ring
    k      : thermal conductivity [W m⁻¹ K⁻¹]
    r_outer: optional outer radius; if None, infer from mesh
    h      : convection coefficient at outer rim [W m⁻² K⁻¹]
    T_inf  : ambient reference temperature [K]

    Returns
    -------
    FiPy CellVariable – steady-state temperature field (K)
    """
    # ---------------- radii ----------------
    dr = mesh.dx
    r_cell = mesh.cellCenters[0].value
    r_inner = float(r_cell.min())
    if r_outer is None:
        r_outer = float(r_cell.max() + dr / 2)

    # ---------------- variable -------------
    T = fp.CellVariable(mesh=mesh, name="temperature", value=T_inf)

    # ---------------- inner Neumann --------
    # −k ∂T/∂r = q_inner  ⇒  ∂T/∂r = −q_inner / k
    T.faceGrad.constrain((-q_inner / k,), where=mesh.facesLeft)

    # ---------------- diffusion term -------
    eq = fp.DiffusionTerm(coeff=k)  # 1/r d/dr(r k dT/dr) in 1-D cylindrical

    # ---------------- outer Robin ----------
    # Build a cell-mask that is 1 only for the last cell
    beta = fp.CellVariable(mesh=mesh, value=0.0)
    beta[-1] = 1.0

    # implicit:  h · T
    eq += fp.ImplicitSourceTerm(coeff=(h / k) * beta)
    # explicit: −h · T_inf
    eq += fp.ExplicitSourceTerm(coeff=(h * T_inf / k) * beta)

    # ---------------- solve ----------------
    eq.solve(var=T)
    return T
