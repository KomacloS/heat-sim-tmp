# laserpad/geometry.py

import fipy as fp

def build_mesh(r_inner: float, r_outer: float, n_r: int = 100) -> fp.Grid1D:
    """Build a 1D radial mesh from r_inner to r_outer with n_r cells."""
    # Compute radial cell size
    dr = (r_outer - r_inner) / n_r

    # Create a simple 1D mesh starting at x=0 (cell centers at dr/2, 3dr/2, …)
    mesh = fp.Grid1D(nx=n_r, dx=dr)

    # Remember absolute radii for the solver
    mesh._r_inner = r_inner
    mesh._r_outer = r_outer

    return mesh



