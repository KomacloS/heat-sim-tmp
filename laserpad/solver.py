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
    """Analytic steady-state with inner Neumann and outer Robin BC (mm units)."""
    # Radii in mm -------------------------------------------------------------
    r_cell = mesh.cellCenters[0].value.copy()
    dr = mesh.dx

    r_face_in = float(r_cell[0] - dr / 2)
    r_face_out = float(r_cell[-1] + dr / 2 if r_outer is None else r_outer)
    r_cell_out = float(r_cell[-1])

    # Constants ---------------------------------------------------------------
    A = -q_inner * r_face_in / k
    B = T_inf + (q_inner * r_face_in) / (h * r_face_out) - A * np.log(r_cell_out)

    # Temperature at cell centres --------------------------------------------
    T_vals = A * np.log(r_cell) + B

    # Store in FiPy variable --------------------------------------------------
    temperature = fp.CellVariable(mesh=mesh, name="temperature")
    temperature[:] = T_vals  # explicit slice assignment populates values

    return temperature
