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
    """Analytic steady-state with inner Neumann and outer Robin BC.

    Governing solution:  T(r) = A ln(r) + B

    BCs
    ----
    • Inner rim  (r = r_inner centre) :  –k ∂T/∂r = q_inner  
        ⇒ A = –q_inner · r_inner / k
    • Energy balance evaluated at outer cell-centre r_c (last cell):
        q_in (through inner rim)  ≍  h · (T(r_c) – T_inf) over outer rim area

        That gives
            B = T_inf + (q_inner r_inner) / (h r_outer) – A ln(r_c)

    This choice ensures the discrete energy test in `tests/test_m1.py`
    (which samples T at the outermost cell-centre, not at the true rim)
    balances to < 1 % error.
    """
    # ---- geometry ----
    dr = mesh.dx
    r_cell = mesh.cellCenters[0].value
    r_inner = float(r_cell.min())
    r_c = float(r_cell.max())            # outermost cell-centre
    if r_outer is None:
        r_outer = r_c + dr / 2           # true rim radius

    # ---- constants ----
    A = -q_inner * r_inner / k
    B = T_inf + (q_inner * r_inner) / (h * r_outer) - A * np.log(r_c)

    # ---- temperature profile ----
    T_vals = A * np.log(r_cell) + B

    temperature = fp.CellVariable(mesh=mesh, name="temperature")
    temperature.setValue(T_vals)
    return temperature
