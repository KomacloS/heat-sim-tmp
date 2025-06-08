"""Simple NumPy-based radial mesh utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    """Uniform 1D radial mesh."""

    r: np.ndarray
    dr: float

    @property
    def r_inner(self) -> float:
        return float(self.r[0] - self.dr / 2)

    @property
    def r_outer(self) -> float:
        return float(self.r[-1] + self.dr / 2)


def build_mesh(r_inner: float, r_outer: float, n_r: int = 100) -> Mesh:
    """Build a uniform radial mesh.

    Parameters
    ----------
    r_inner:
        Inner radius.
    r_outer:
        Outer radius.
    n_r:
        Number of cells.

    Returns
    -------
    Mesh
        Mesh dataclass containing cell centres and spacing.
    """

    dr = (r_outer - r_inner) / n_r
    r = np.linspace(r_inner + dr / 2, r_outer - dr / 2, n_r)
    return Mesh(r=r, dr=dr)

