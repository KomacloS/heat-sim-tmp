# laserpad/geometry.py
import fipy as fp

def build_mesh(r_inner: float, r_outer: float, n_r: int = 100) -> fp.Grid1D:
    """Build a 1D radial mesh with absolute radii (in metres)."""
    dr = (r_outer - r_inner) / n_r
    mesh = fp.Grid1D(nx=n_r, dx=dr)
    mesh.faceCenters[0].value[:] += r_inner
    mesh.cellCenters[0].value[:] += r_inner
    return mesh
