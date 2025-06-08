import numpy as np


def build_mesh(r_inner: float, r_outer: float, n_r: int = 100) -> tuple[np.ndarray, float]:
    """Return radial cell centres and cell width for a 1D ring.

    Parameters
    ----------
    r_inner:
        Inner radius of the ring in metres.
    r_outer:
        Outer radius of the ring in metres.
    n_r:
        Number of radial cells.

    Returns
    -------
    tuple[np.ndarray, float]
        ``r_centres`` array of length ``n_r`` giving the radial cell centres in
        metres and ``dr`` the uniform cell width in metres.
    """

    dr = (r_outer - r_inner) / n_r
    r_centres = r_inner + (np.arange(n_r) + 0.5) * dr

    return r_centres, dr
