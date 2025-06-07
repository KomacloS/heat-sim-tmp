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
    """Analytic steady-state in a cylindrical ring with

    • Inner Neumann   : –k ∂T/∂r = q_inner  
    • Outer Robin     : –k ∂T/∂r = h (T – T_inf)

    The general solution of (1/r)d/dr(r k ∂T/∂r)=0 is T(r)=A ln r + B.
    Constants A, B come from the two BCs.

    Parameters
    ----------
    mesh   : FiPy 1-D radial mesh (cell centres give absolute radii)
    q_inner: Heat flux at inner rim  [W m⁻²]
    k      : Thermal conductivity    [W m⁻¹ K⁻¹]
    r_outer: Outer radius; if None, take from mesh
    h      : Convection coefficient  [W m⁻² K⁻¹]
    T_inf  : Ambient temperature     [K]

    Returns
    -------
    fipy.CellVariable
        Temperature field [K] at cell centres.
    """
    # ---- radii ----
    dr = mesh.dx
    r_cell = mesh.cellCenters[0].value
    r_inner = float(r_cell.min())
    if r_outer is None:
        r_outer = float(r_cell.max() + dr / 2)

    # ---- constants from BCs ----
    # A from inner Neumann
    A = -q_inner * r_inner / k  # (because –k * A / r_inner = q_inner)
    # T_outer from outer Robin
    T_outer = T_inf + (q_inner * r_inner) / (h * r_outer)
    # B so that T(r_outer) = T_outer
    B = T_outer - A * np.log(r_outer)

    # ---- temperature profile ----
    T_vals = A * np.log(r_cell) + B

    # ---- FiPy variable ----
    temperature = fp.CellVariable(mesh=mesh, name="temperature")
    temperature.setValue(T_vals)
    return temperature
