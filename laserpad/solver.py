# laserpad/solver.py

import fipy as fp

def solve_steady(
    mesh: fp.Grid1D, q_inner: float, k: float = 400.0
) -> fp.CellVariable:
    """Solve 1D steady‐state cylindrical conduction with inner heat‐flux and insulated outer rim.

    Equation: (1/r) d/dr (r k dT/dr) = 0
    BCs:  -k dT/dr |_{inner} = q_inner,    dT/dr |_{outer} = 0
    """
    # Initialize temperature field (initial value doesn't impose Dirichlet on outer)
    T = fp.CellVariable(mesh=mesh, name="temperature", value=0.0)

    # Build the axisymmetric diffusion equation
    eq = fp.DiffusionTerm(coeff=k)

    # Apply a Neumann BC at the inner face (mesh.facesLeft):
    #   -k dT/dr = q_inner  ⇒  add (q_inner / k) at that face
    eq += (q_inner / k) * mesh.facesLeft

    # Solve steady‐state (outer default is zero‐flux)
    eq.solve(var=T)

    return T






