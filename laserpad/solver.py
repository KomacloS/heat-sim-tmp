from typing import Optional

import fipy as fp


def solve_steady(
    mesh: fp.Grid1D,
    q_inner: float,
    k: float = 400.0,
    r_outer: Optional[float] = None,  # not needed but kept for API compat
    h: float = 1_000.0,
    T_inf: float = 0.0,
) -> fp.CellVariable:
    """Steady-state cylindrical conduction solved with FiPy FVM (SI units).

    BCs
    ----
    • Inner face (r = r_inner) :  −k ∂T/∂r = q_inner        (Neumann)
    • Outer face (r = r_outer) :  −k ∂T/∂r = h (T − T_inf)  (Robin)

    The Robin BC is enforced by constraining the face gradient:
        ∂T/∂r |_{outer} = −(h / k) · (T_face − T_inf)
    """
    # ---- Variable, initialised to T_inf everywhere ----
    T = fp.CellVariable(mesh=mesh, value=T_inf, name="temperature")

    # ---- Inner Neumann heat-flux ----
    #     −k ∂T/∂r = q_inner  ⇒  ∂T/∂r = −q_inner / k
    T.faceGrad.constrain((-q_inner / k,), where=mesh.facesLeft)

    # ---- Outer Robin convection ----
    #     ∂T/∂r = −(h/k)(T − T_inf)  (applied on outer faces)
    alpha = h / k
    T.faceGrad.constrain((-alpha) * (T.faceValue - T_inf), where=mesh.facesRight)

    # ---- Cylindrical diffusion equation ----
    eq = fp.DiffusionTerm(coeff=k)  # FiPy uses (1/r)∂/∂r(r k ∂T/∂r) in 1-D

    # ---- Solve steady-state ----
    eq.solve(var=T)
    return T
