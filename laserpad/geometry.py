import fipy as fp

def build_mesh(r_inner: float, r_outer: float, n_r: int = 100) -> fp.Grid1D:
    """Build a 1D radial mesh from r_inner to r_outer with n_r cells.

    After creating a 0→(r_outer−r_inner) mesh, shift both face- and
    cell-centres by +r_inner so they represent absolute radii.
    """
    dr = (r_outer - r_inner) / n_r
    mesh = fp.Grid1D(nx=n_r, dx=dr)

    # Shift face-centres (boundaries) to [r_inner, r_outer]
    mesh.faceCenters[0].value[:] += r_inner
    # Shift cell-centres to [r_inner+dr/2, …, r_outer−dr/2]
    mesh.cellCenters[0].value[:] += r_inner

    return mesh
