# laserpad/geometry.py

import fipy as fp
from typing import Tuple

def build_mesh(r_inner: float, r_outer: float, n_r: int = 100) -> fp.Grid1D:
    """Build a 1D radial mesh from r_inner to r_outer with n_r cells."""
    dr = (r_outer - r_inner) / n_r
    mesh = fp.Grid1D(nx=n_r, dx=dr)          # no origin argument!
    mesh._r_inner = r_inner                  # stash true radii
    mesh._r_outer = r_outer
    return mesh

