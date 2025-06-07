import math
from typing import Optional

import fipy as fp


def solve_steady(
    mesh: fp.Grid1D,
    q_inner: float,
    k: float = 400.0,
    r_outer: Optional[float] = None,
    h: float = 1_000.0,
    T_inf: float = 0.0,
) -> fp.CellVariable:
    """Steady-state cylindrical conduction solved numerically with FiPy FVM.

    BCs
    ----
    • Inner rim  (r = r_inner)  :  −k ∂T/∂r = q_inner   (Neumann)  
    • Outer rim  (r = r_outer)  :  −k ∂T/∂r = h (T − T_inf)  (Robin)

    Parameters
    ----------
    mesh
        1-D radial `Grid1D`; cell centres hold absolute radii (m).
    q_inner
        Applied heat flux at the inner rim (W m⁻², positive into domain).
    k
        Thermal conductivity of copper (W m⁻¹ K⁻¹).
    r_outer
        Optional outer radius; if `None`, computed from mesh.
    h
        Convection coefficient at the outer rim (W m⁻² K⁻¹).
    T_inf
        Ambient reference temperature for convection (K).

    Returns
    -------
    fipy.CellVariable
        Steady-state temperature field (K).
    """
    # ---------------- absolute radii ----------------
    dr = mesh.dx
    r_cell = mesh.cellCenters[0].value
    r_inner = float(r_cell.min())
    if r_outer is None:
        r_outer = float(r_cell.max() + dr / 2)

    # ---------------- variable ----------------------
    T = fp.CellVariable(mesh=mesh, name="temperature", value=T_inf)

    # ---------------- inner Neumann -----------------
    # FiPy: impose gradient directly on faces
    #   −k ∂T/∂r = q_inner  ⇒  ∂T/∂r = −q_inner / k
    T.faceGrad.constrain((-q_inner / k,), where=mesh.facesLeft)

    # ---------------- governing equation ------------
    # DiffusionTerm in 1-D uses axisymmetric form (1/r) d/dr(r k dT/dr)
    eq = fp.DiffusionTerm(coeff=k)

    # ---------------- outer Robin -------------------
    #   −k ∂T/∂r = h (T − T_inf)
    # Implemented via implicit + explicit source terms
    outer = mesh.facesRight

    eq += fp.ImplicitSourceTerm(coeff=h / k, var=T, where=outer)  # h·T
    eq += fp.ExplicitSourceTerm(coeff=(h * T_inf) / k, where=outer)  # −h·T_inf

    # ---------------- solve -------------------------
    eq.solve(var=T)
    return T
