# laserpad/geometry.py

import fipy as fp
from typing import Tuple

def build_mesh(r_inner: float, r_outer: float, n_r: int = 100) -> fp.Grid1D:
    """Build a 1D radial mesh from r_inner to r_outer with n_r cells."""
    # Compute radial cell size
    dr = (r_outer - r_inner) / n_r

    # Create a 1D FiPy mesh with cell centers at:
    #   r = r_inner + (i + 0.5)*dr  for i in [0..n_r-1]
    mesh = fp.Grid1D(nx=n_r, dx=dr, origin=r_inner + dr/2)

    return mesh

