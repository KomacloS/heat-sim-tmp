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
    """Numerical steady-state cylindrical conduction with

    * inner **Neumann** heat-flux  –k ∂T/∂r = q_inner
    * outer **Robin** convection  –k ∂T/∂r = h (T – T_inf)

    Parameters
    ----------
    mesh
        1-D FiPy mesh whose cell centres already hold absolute radii [m].
    q_inner
        Applied heat flux [W m⁻²] at *r_inner*.
    k
        Thermal conductivity of copper [W m⁻¹ K⁻¹].
    r_outer
        Optional outer radius (defaults to last cell centre + dx/2).
    h
        Heat-transfer coefficient at outer rim [W m⁻² K⁻¹].  **Default = 1000**.
    T_inf
        Ambient reference temperature at outer rim [K].  **Default = 0**.

    Returns
    -------
    fipy.CellVariable
        Temperature field [K] at cell centres.
    """
    # ------------------------------------------------------------------ mesh info
    dr = mesh.dx
    r_cell = mesh.cellCenters[0].value
    r_inner = float(r_cell.min())
    if r_outer is None:
        r_outer = float(r_cell.max() + dr / 2)

    # ------------------------------------------------------------------ variable
    T = fp.CellVariable(mesh=mesh, name="temperature", value=T_inf)

    # ------------------------------------------------------------------ governing eq
    eq = fp.DiffusionTerm(coeff=k)  # FiPy in 1-D uses axisymmetric form automatically

    # -------- inner Neumann (heat-flux into domain)
    eq += (q_inner / k) * mesh.facesLeft

    # -------- outer Robin convection
    outer = mesh.facesRight
    eq += fp.ImplicitSourceTerm(coeff=h / k, var=T, where=outer)  # h*(T - T_inf)
    eq += (h * T_inf / k) * outer                                # adds +h*T_inf

    # ------------------------------------------------------------------ solve
    eq.solve(var=T)
    return T
