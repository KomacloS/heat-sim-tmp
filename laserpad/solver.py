from typing import Optional

import fipy as fp


def solve_steady(
    mesh: fp.Grid1D,
    q_inner: float,
    k: float = 400.0,
    r_outer: Optional[float] = None,  # kept for API compatibility
    h: float = 1_000.0,
    T_inf: float = 0.0,
) -> fp.CellVariable:
    """Steady-state cylindrical conduction (FiPy FVM, SI units).

    BCs
    ----
    • Inner face  : −k ∂T/∂r = q_inner        (Neumann)
    • Outer face  : −k ∂T/∂r = h (T − T_inf)  (Robin)

    Implementation
    --------------
    * Inner flux is imposed with `faceGrad.constrain`.
    * Robin BC is converted to an *equivalent Dirichlet* temperature on the
      outer faces using the ghost-cell formula:

        T_face = (k · T_cell + h · Δr/2 · T_inf) / (k + h · Δr/2)

      where Δr/2 is the half-cell distance from the last cell centre to the
      boundary.  Constraining `faceValue` like this makes the FiPy matrix
      nonsingular and fully implicit.
    """
    dr = float(mesh.dx)                     # [m]
    half_dx = dr / 2.0

    # ------------------------------------------------------------------ var
    T = fp.CellVariable(mesh=mesh, value=T_inf, name="temperature")

    # ------------------------------------------------------------------ inner Neumann
    T.faceGrad.constrain((-q_inner / k,), where=mesh.facesLeft)

    # ------------------------------------------------------------------ outer Robin as Dirichlet
    # For the last *cell* value T_cell_out, the equivalent boundary face value:
    #     T_face = (k T_cell_out + h Δr/2 T_inf) / (k + h Δr/2)
    coeff = h * half_dx / (k + h * half_dx)           #  h Δr/2 / (k + h Δr/2)
    T_face_expr = (1.0 - coeff) * T + coeff * T_inf   # linear in T (implicit)
    T.faceValue.constrain(T_face_expr, where=mesh.facesRight)

    # ------------------------------------------------------------------ diffusion eqn
    eq = fp.DiffusionTerm(coeff=k)

    # ------------------------------------------------------------------ solve
    eq.solve(var=T)
    return T
