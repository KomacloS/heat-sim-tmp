# laserpad/geometry.py

import fipy as fp

def build_mesh(r_inner: float, r_outer: float, n_r: int = 100) -> fp.Grid1D:
    """Build a 1D radial mesh from r_inner to r_outer with n_r cells."""
    dr = (r_outer - r_inner) / n_r
    # Place cell‐centers at absolute radii: r_inner + (i+0.5)*dr
    mesh = fp.Grid1D(nx=n_r, dx=dr, origin=r_inner + dr/2)
    return mesh


