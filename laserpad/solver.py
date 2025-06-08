"""Analytic steady-state solver for radial conduction."""

from __future__ import annotations

import numpy as np


def solve_steady(
    r: np.ndarray,
    dr: float,
    q_inner: float,
    k: float = 400.0,
    r_outer: float | None = None,
    h: float = 1_000.0,
    T_inf: float = 0.0,
    r_inner: float | None = None,
) -> np.ndarray:
    """Return the steady-state temperature profile on ``r``.

    The governing equation is the axisymmetric steady conduction problem with a
    prescribed heat flux ``q_inner`` at ``r_inner`` and a convective boundary at
    ``r_outer``. The solution is derived analytically.
    """
    if r_inner is None:
        r_inner = float(r[0] - dr / 2)
    if r_outer is None:
        r_outer = float(r[-1] + dr / 2)
    # Analytic solution with convection imposed at ``r_outer``
    # -------------------------------------------------------
    # For a radial domain r_inner <= r <= r_outer with flux ``q_inner`` entering
    # at the inner boundary and convection to ``T_inf`` at the outer boundary,
    # the temperature profile is
    #
    #   T(r) = B - (q_inner * r_inner / k) * ln(r)
    #   B = T_inf + (q_inner * r_inner) / (h * r_outer)
    #       + (q_inner * r_inner / k) * ln(r_outer)
    #
    # The above ensures heat balance: the outer face temperature satisfies
    # ``-k dT/dr(r_outer) = h (T(r_outer) - T_inf)``.
    coeff = q_inner * r_inner / k
    B = T_inf + coeff * np.log(r_outer) + (q_inner * r_inner) / (h * r_outer)
    values = B - coeff * np.log(r)

    return values
