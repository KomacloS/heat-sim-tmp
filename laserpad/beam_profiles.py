"""Beam profile factories for distributed heat sources."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import cast


def uniform_beam(r: NDArray[np.float_], q0: float) -> NDArray[np.float_]:
    """Return constant heat flux q'' = q0 across all radii."""
    return cast(NDArray[np.float_], np.full_like(r, q0, dtype=float))


def gaussian_beam(
    r: NDArray[np.float_], peak_q: float, sigma: float
) -> NDArray[np.float_]:
    """Return a Gaussian heat-flux profile."""
    return cast(NDArray[np.float_], peak_q * np.exp(-0.5 * (r / sigma) ** 2))


def donut_beam(
    r: NDArray[np.float_], inner_r: float, outer_r: float, q0: float
) -> NDArray[np.float_]:
    """Return a simple ring-shaped heat-flux profile."""
    profile: NDArray[np.float_] = np.zeros_like(r, dtype=float)
    mask = (r >= inner_r) & (r <= outer_r)
    profile[mask] = q0
    return profile
