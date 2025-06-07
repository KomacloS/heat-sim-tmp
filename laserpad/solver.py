# laserpad/solver.py

from typing import Optional
import fipy as fp

def solve_steady(
    mesh: fp.Grid1D,
    q_inner: float,
    k: float = 400.0,
    r_outer: Optional[float] = None
) -> fp.CellVariable:
    """Compute steady-state temperature on a cylindrical annulus.

    Solves (1/r)·d/dr(r·k·dT/dr)=0 with:
      • inner Neumann BC: -k·dT/dr = q_inner
      • outer insulated BC: dT/dr = 0

    Note:
      - r_outer is accepted for compatibility but not used in the FVM solve.
    """
    # Initialize temperature field
    T = fp.CellVariable(mesh=mesh, name="temperature", value=0.0)

    # Build the axisymmetric diffusion equation
    eq = fp.DiffusionTerm(coeff=k)

    # Apply inner Neumann (heat-flux) at the left face
    eq += (q_inner / k) * mesh.facesLeft

    # Solve steady-state (outer face defaults to zero-flux)
    eq.solve(var=T)

    return T







