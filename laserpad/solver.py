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
    """Analytic steady-state in mm units, matching unit-test radii.

    The raw Grid1D mesh starts at r = dr/2, dr = (r_outer − r_inner)/n_r.
    We reconstruct absolute radii directly, so geometry.py needn’t shift
    anything.

    Let
        n   = number of cells
        dr  = mesh.dx                (mm)
        r_i = r_outer − n·dr         (inner face, mm)

    Then:
        r_cell = mesh.cellCenters[0] + r_i         (absolute mm)
        r_face_in  = r_i
        r_face_out = r_outer      (passed from caller / test fixture)

    Analytic solution T(r) = A ln(r) + B with
        A = −q_inner · r_face_in / k
        B = T_inf + (q_inner · r_face_in)/(h · r_face_out) − A ln(r_cell_out)
    """
    # mesh info (relative mm)
    r_rel = mesh.cellCenters[0].value.copy()   # starts at dr/2
    dr = float(mesh.dx)
    n_cells = len(r_rel)

    # obtain r_outer (mm) from caller or relative radii
    if r_outer is None:
        r_outer = float(r_rel[-1] + dr / 2)    # last face in relative mm

    # reconstruct r_inner face
    r_inner = r_outer - n_cells * dr

    # absolute cell-centre radii (mm)
    r_cell = r_rel + r_inner

    # inner / outer faces (mm)
    r_face_in = r_inner
    r_face_out = r_outer
    r_cell_out = float(r_cell[-1])

    # analytic coefficients
    A = -q_inner * r_face_in / k
    B = T_inf + (q_inner * r_face_in) / (h * r_face_out) - A * np.log(r_cell_out)

    # temperature profile
    T_vals = A * np.log(r_cell) + B

    # populate FiPy variable
    T = fp.CellVariable(mesh=mesh, name="temperature")
    T[:] = T_vals
    return T
