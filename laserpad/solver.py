# laserpad/solver.py
from typing import Optional
import fipy as fp

def solve_steady(
    mesh: fp.Grid1D,
    q_inner: float,
    k: float = 400.0,
    r_outer: Optional[float] = None,
    h: float = 1e3,
    T_inf: float = 0.0
) -> fp.CellVariable:
    """FiPy FVM steady-state solve with Neumann inner and Robin outer BC."""
    # Initialize T
    T = fp.CellVariable(mesh=mesh, name="temperature", value=T_inf)

    # Inner Neumann: -k dT/dr = q_inner  ⇒  dT/dr = -q_inner/k
    T.faceGrad.constrain((-q_inner / k,), where=mesh.facesLeft)

    # Build diffusion equation (cylindrical form)
    eq = fp.DiffusionTerm(coeff=k)

    # Outer Robin: -k dT/dr = h (T - T_inf)
    # Implicit source: + (h/k)*T on outer faces
    eq += fp.ImplicitSourceTerm(coeff=h / k, var=T, where=mesh.facesRight)
    # Explicit source: - (h*T_inf/k) on outer faces
    eq += fp.ExplicitSourceTerm(coeff=-(h * T_inf / k), where=mesh.facesRight)

    # Solve
    eq.solve(var=T)
    return T
    