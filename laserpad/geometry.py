import fipy as fp

def build_mesh(r_inner: float, r_outer: float, n_r: int = 100) -> fp.Grid1D:
    """Build a 1D radial mesh from r_inner to r_outer with n_r cells.

    After creating a mesh on [0…r_outer−r_inner], shift both face- and
    cell-centres by +r_inner so they represent absolute radii in mm.
    """
    dr = (r_outer - r_inner) / n_r
    mesh = fp.Grid1D(nx=n_r, dx=dr)

    # Shift face-centres to [r_inner, r_outer]
    mesh.faceCenters[0].value[:] += r_inner
    # Shift cell-centres to [r_inner + dr/2 … r_outer − dr/2]
    mesh.cellCenters[0].value[:] += r_inner

    return mesh
