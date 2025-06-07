# laserpad/geometry.py

import fipy as fp

def build_mesh(r_inner: float, r_outer: float, n_r: int = 100) -> fp.Grid1D:
    """Build a 1D cylindrical mesh with absolute radii in the cell‐centers."""
    # Cell width
    dr = (r_outer - r_inner) / n_r

    # Base mesh runs from 0 → (r_outer - r_inner)
    mesh = fp.Grid1D(nx=n_r, dx=dr)

    # Shift the mesh so that cellCenters actually span [r_inner+dr/2 … r_outer−dr/2]:
    # This makes mesh.cellCenters[0] already be the physical radius.
    mesh.faceCenters[0].value[:] += r_inner
    mesh.cellCenters[0].value[:] += r_inner

    return mesh




