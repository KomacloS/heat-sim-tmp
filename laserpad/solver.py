import fipy as fp
import numpy as np

def solve_steady(
    mesh,
    q_inner,
    k: float = 400.0,
    r_outer=None,     # <— add this back so test’s r_outer=... won’t blow up
    h: float = 1_000.0,
    T_inf: float = 0.0,
):
    """
    Steady 1-D conduction in a ring (r-direction only, SI units).

    Inner face:  −k ∂T/∂r =  q_inner          (Neumann)
    Outer face:  −k ∂T/∂r =  h (T − T_inf)    (Robin)
    """
    # ------------------------------------------------------------------ unknown
    T = fp.CellVariable(mesh=mesh, name="temperature", value=T_inf)

    # ---------------------------------------------------------------- inner BC
    T.faceGrad.constrain((-q_inner / k,), where=mesh.facesLeft)

    # ---------------------------------------------------------------- outer BC
    # Robin:  ∂T/∂r = −h/k (T − T_inf)   (imposed on outer face)
    T.faceGrad.constrain(
        (-h / k) * (T.faceValue - T_inf), where=mesh.facesRight
    )

    # ---------------------------------------------------------------- operator
    r = mesh.cellCenters[0]              # absolute radii [m]

    # For steady, source-free radial conduction:
    #     d/dr ( r k dT/dr ) = 0
    # in FiPy form → DiffusionTerm( k * r )
    eq = fp.DiffusionTerm(coeff=k * r, var=T)

    eq.solve(var=T)
    return T
