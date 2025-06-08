"""Steady-state solver for radial conduction using NumPy."""

from __future__ import annotations

import numpy as np


def solve_steady(
    r_centres: np.ndarray,
    dr: np.ndarray,
    q_inner: float,
    k: float = 400.0,
    r_outer: float | None = None,
    h: float = 1_000.0,
    T_inf: float = 0.0,
    r_inner: float | None = None,
) -> np.ndarray:
    """Return the steady-state temperature profile on the radial mesh."""
    if r_inner is None:
        r_inner = float(r_centres[0] - dr[0] / 2)
    if r_outer is None:
        r_outer = float(r_centres[-1] + dr[-1] / 2)

    coeff = q_inner * r_inner / k
    B = T_inf + coeff * np.log(r_outer) + (q_inner * r_inner) / (h * r_outer)
    values = B - coeff * np.log(r_centres)
    return values
