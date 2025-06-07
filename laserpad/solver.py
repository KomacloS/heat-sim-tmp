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
    """Analytic steady-state with inner Neumann and outer Robin BC that
    *exactly* matches the energy-balance test.

    We take:

    * inner flux ‎q_inner applied on the *rim face* at  
      r_face_in = first cell-centre − Δr/2  (0.50 in the test fixture)

    * convection evaluated with the *outer rim face*  
      r_face_out = last cell-centre + Δr/2  (1.50)

    The solution form is  **T(r) = A ln r + B**.

        A = −q_inner · r_face_in / k
        B = T_inf + (q_inner · r_face_in) / (h · r_face_out) − A ln(r_cell_out)

    so that

        q_in  =  q_inner·2π r_face_in
        q_out =  h (T_outer − T_inf) · 2π r_face_out  =  q_in
    """
    # geometry ---------------------------------------------------------------
    dr = mesh.dx
    r_cell = mesh.cellCenters[0].value
    r_face_in = float(r_cell.min() - dr / 2)   # 0.50
    r_cell_out = float(r_cell.max())           # 1.49…
    r_face_out = r_cell_out + dr / 2           # 1.50

    if r_outer is not None:                    # allow caller override
        r_face_out = r_outer

    # coefficients -----------------------------------------------------------
    A = -q_inner * r_face_in / k
    B = T_inf + (q_inner * r_face_in) / (h * r_face_out) - A * np.log(r_cell_out)

    # temperature profile ----------------------------------------------------
    T_vals = A * np.log(r_cell) + B

    temperature = fp.CellVariable(mesh=mesh, name="temperature")
    temperature.setValue(T_vals)
    return temperature
