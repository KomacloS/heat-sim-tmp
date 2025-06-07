# laserpad/geometry.py

import fipy as fp
from typing import Tuple

def build_mesh(r_inner: float, r_outer: float, n_r: int = 100) -> fp.Grid1D:
    """Build a 1D radial mesh from r_inner to r_outer with n_r cells."""
    # Compute the radial cell width
    dr = (r_outer - r_inner) / n_r

    # Create a simple 1D mesh (length = r_outer - r_inner)
    mesh = fp.Grid1D(nx=n_r, dx=dr)

    # Stash the true radii on the mesh object so the solver can recover absolute positions
    mesh._r_inner = r_inner
    mesh._r_outer = r_outer

    return mesh
