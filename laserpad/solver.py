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
    """Analytic steady-state in mm‐units matching test harness.

    BCs (using mm for radii):
      –k ∂T/∂r = q_inner      on inner face at r_face_in [mm]
      –k ∂T/∂r = h (T – T_inf) on outer face at r_face_out [mm]

    Solution form: T(r) = A·ln(r) + B, with
        A = –q_inner·r_face_in / k
        B = T_inf + (q_inner·r_face_in)/(h·r_face_out) – A·ln(r_face_out)
    """
    # fetch mm‐units radii
    r_face = mesh.faceCenters[0].value.copy()   # in mm
    r_cell = mesh.cellCenters[0].value.copy()   # in mm

    r_face_in = float(r_face[0])
    r_face_out = float(r_outer if r_outer is not None else r_face[-1])
    r_cell_out = float(r_cell[-1])

    # compute constants
    A = -q_inner * r_face_in / k
    B = T_inf + (q_inner * r_face_in) / (h * r_face_out) - A * np.log(r_face_out)

    # temperature at each cell‐centre
    T_vals = A * np.log(r_cell) + B

    T = fp.CellVariable(mesh=mesh, name="temperature")
    T.setValue(T_vals)
    return T
