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
    """Analytic steady-state with inner Neumann and outer Robin BC in SI units.

    Mesh radii are supplied in mm, so we convert to metres:
      r_m = r_mm * 1e-3.

    BCs:
      –k ∂T/∂r = q_inner   at inner rim face
      –k ∂T/∂r = h (T – T_inf)   at outer rim face
    """
    # ---- read and convert radii from mm → m ----
    dr_mm = mesh.dx  # in mm
    r_face_mm = mesh.faceCenters[0].value
    r_cell_mm = mesh.cellCenters[0].value

    # convert to metres
    dr = dr_mm * 1e-3
    r_face_m = r_face_mm * 1e-3
    r_cell_m = r_cell_mm * 1e-3

    # identify inner/outer face and cell-centre radii
    r_face_in = float(r_face_m[0])
    r_face_out = float(r_face_m[-1])
    r_cell_out = float(r_cell_m[-1])

    # override if caller passed r_outer (mm → m)
    if r_outer is not None:
        r_face_out = r_outer * 1e-3

    # ---- compute analytic constants ----
    A = -q_inner * r_face_in / k
    B = T_inf + (q_inner * r_face_in) / (h * r_face_out) - A * np.log(r_cell_out)

    # ---- evaluate temperature at each cell-centre (in K) ----
    T_vals = A * np.log(r_cell_m) + B

    temperature = fp.CellVariable(mesh=mesh, name="temperature")
    temperature.setValue(T_vals)
    return temperature
