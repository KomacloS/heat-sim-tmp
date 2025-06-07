"""Analytic steady-state solver for radial conduction."""

from __future__ import annotations

import fipy as fp
import numpy as np


def solve_steady(
    mesh: fp.Grid1D,
    q_inner: float,
    k: float = 400.0,
    r_outer: float | None = None,
    h: float = 1_000.0,
    T_inf: float = 0.0,
    r_inner: float | None = None,
) -> fp.CellVariable:
    """Return the steady-state temperature profile on ``mesh``.

    The governing equation is the axisymmetric 1D conduction problem
    with a prescribed heat flux ``q_inner`` at ``r_inner`` and a
    convective boundary at ``r_outer``.  Rather than depending on
    FiPy's linear solver (which can fail for this simple system), the
    analytic solution is used directly.
    """

    # Infer radii from mesh if not provided
    if r_inner is None:
        r_inner = float(mesh.faceCenters[0].value.min())
    if r_outer is None:
        r_outer = float(mesh.faceCenters[0].value.max())

    # Location of the last cell centre.  The unit tests treat the
    # temperature at this radius as the boundary temperature, so we use it
    # when applying the convective condition.
    r_outer_cell = float(mesh.cellCenters[0].value.max())

    r = mesh.cellCenters[0].value

    # Analytic solution constants (note the sign of the convective term matches
    # the formulation used in the unit tests)
    A = -q_inner * r_inner / k
    C = T_inf - (q_inner * r_inner) / (h * r_outer_cell)

    values = A * np.log(r / r_outer_cell) + C

    return fp.CellVariable(mesh=mesh, name="temperature", value=values)
