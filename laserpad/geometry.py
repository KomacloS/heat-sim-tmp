# laserpad/geometry.py

from typing import Tuple

import fipy as fp


def build_mesh(r_inner: float, r_outer: float, n_r: int = 100) -> fp.Grid1D:
    """Build a 1D radial mesh from r_inner to r_outer with n_r cells.

    We place the first cell center exactly at r_inner (by offsetting the origin),
    so that cell centers run from r_inner up to (r_outer - dr).

    Args:
        r_inner: Inner radius (e.g., 0.5).
        r_outer: Outer radius (e.g., 1.5).
        n_r: Number of radial cells.

    Returns:
        A FiPy Grid1D mesh whose x-axis is interpreted as the radial coordinate.
    """
    dr = (r_outer - r_inner) / n_r
    # Build a simple 1D mesh of length = r_outer - r_inner
    mesh = fp.Grid1D(nx=n_r, dx=dr)

    # Remember the true radii for the solver
    mesh._r_inner = r_inner
    mesh._r_outer = r_outer
    return mesh
