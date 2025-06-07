import fipy as fp

def build_mesh(r_inner: float, r_outer: float, n_r: int = 100) -> fp.Grid1D:   

    dr = (r_outer - r_inner) / n_r    
    # Build a uniform mesh spanning ``[0, r_outer - r_inner]`` then translate
    # it so that the first face coincides with ``r_inner``.
    base = fp.Grid1D(nx=n_r, dx=dr)
    mesh = base + (r_inner,)

    return mesh