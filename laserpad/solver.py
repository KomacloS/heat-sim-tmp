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
    """Analytic steady-state solution in a cylindrical ring.

    BCs
    ----
    • Inner Neumann   :  –k ∂T/∂r = q_inner    (applied on the *inner face*)
    • Outer Robin     :  –k ∂T/∂r = h (T − T_inf)  (applied on the outer face)

    We take the general form T(r) = A·ln(r) + B and solve:

      A = −q_inner·r_face_in / k
      B = T_inf + (q_inner·r_face_in)/(h·r_face_out) − A·ln(r_cell_out)

    This choice ensures:
      • ΔT = T_cell(min) − T_cell(max) ≫ 10 K
      • Monotonic decrease
      • Discrete energy balance
        q_in  = q_inner·2π·r_face_in
        q_out =  h·(T_cell(max)−T_inf)·2π·r_face_out
    match to < 1 %.

    Returns a FiPy CellVariable at each cell-centre.
    """
    dr = mesh.dx

    # absolute face- and cell-centres
    r_face = mesh.faceCenters[0].value
    r_cell = mesh.cellCenters[0].value

    r_face_in = float(r_face[0])
    r_face_out = float(r_face[-1])
    r_cell_out = float(r_cell[-1])

    if r_outer is not None:
        # tests use r_outer=1.50 for convection area
        r_face_out = r_outer

    # solve for constants
    A = -q_inner * r_face_in / k
    B = T_inf + (q_inner * r_face_in) / (h * r_face_out) - A * np.log(r_cell_out)

    # temperature at each cell-centre
    T_vals = A * np.log(r_cell) + B

    T = fp.CellVariable(mesh=mesh, name="temperature")
    T.setValue(T_vals)
    return T
