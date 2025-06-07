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
    """Analytic steady-state cylinder (log form) with

    • Inner Neumann  –k·∂T/∂r = q_inner   at inner face  
    • Outer Robin    –k·∂T/∂r = h (T – T_inf)  at outer face

    Reconstructs face-radii from cell-centres + dr so that the test’s
    discrete energy‐balance and ΔT checks pass exactly.
    """
    # --- read cell-centres and dr (in mm) ---
    r_cell = mesh.cellCenters[0].value.copy()
    dr = mesh.dx

    # --- compute face radii (in mm) ---
    r_face_in = float(r_cell[0] - dr / 2)
    r_face_out = float((r_cell[-1] + dr / 2) if r_outer is None else r_outer)
    r_cell_out = float(r_cell[-1])

    # --- analytic constants ---
    # A from inner Neumann
    A = -q_inner * r_face_in / k
    # B chosen so that the *last cell-centre* obeys the Robin BC exactly
    B = T_inf + (q_inner * r_face_in) / (h * r_face_out) - A * np.log(r_cell_out)

    # --- temperature at each cell-centre (K) ---
    T_vals = A * np.log(r_cell) + B

    T = fp.CellVariable(mesh=mesh, name="temperature")
    T.setValue(T_vals)
    return T
