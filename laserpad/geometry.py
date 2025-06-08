from __future__ import annotations

import numpy as np


def build_mesh(
    r_inner: float, r_outer: float, n_r: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Return cell centres and cell widths for a 1‑D radial mesh."""

    dr = (r_outer - r_inner) / n_r
    r_edges = np.linspace(r_inner, r_outer, n_r + 1)
    r_centres = (r_edges[:-1] + r_edges[1:]) / 2

    dr_array = np.full_like(r_centres, dr)

    return r_centres, dr_array
