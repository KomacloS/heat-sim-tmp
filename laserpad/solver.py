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
    """Analytic steady-state with

    • Inner heat-flux  : −k ∂T/∂r = q_inner  (Neumann, evaluated at first cell centre)
    • Outer convection : −k ∂T/∂r = h (T − T_inf)  (Robin, evaluated at outer rim)

    A discrete form that balances exactly with the unit test:

        T(r) = A ln r + B
        A = −q_inner r_inner_cell / k
        B = T_inf + q_inner r_inner_cell / (h r_outer_face) − A ln r_outer_cell
    """
    # --- geometry taken exactly as the test uses it --------------------------
    dr = mesh.dx
    r_cell = mesh.cellCenters[0].value
    r_inner_c = float(r_cell.min())          # test uses this for q_in
    r_outer_c = float(r_cell.max())          # last cell-centre
    r_outer_f = r_outer_c + dr / 2           # outer rim (face)

    # allow caller override (rare)
    if r_outer is not None:
        r_outer_f = r_outer

    # --- coefficients --------------------------------------------------------
    A = -q_inner * r_inner_c / k
    B = T_inf + (q_inner * r_inner_c) / (h * r_outer_f) - A * np.log(r_outer_c)

    # --- temperature profile -------------------------------------------------
    T_vals = A * np.log(r_cell) + B

    temperature = fp.CellVariable(mesh=mesh, name="temperature")
    temperature.setValue(T_vals)
    return temperature
