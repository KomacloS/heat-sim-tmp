# laserpad/solver.py
"""
Analytic steady-state temperature in a 1-D copper ring.

We solve
    (1/r) d/dr ( r k dT/dr ) = 0                on  r_inner ≤ r ≤ r_outer
with
    –k (dT/dr)|_r=r_inner = q_inner              (prescribed inward heat-flux)
    –k (dT/dr)|_r=r_outer = h [T – T_inf]        (convective cooling)
and constant k, h.

The solution is

    T(r) = (q_inner r_inner / k) ln(r_outer / r)
           + (q_inner r_inner) / (h r_outer)
           + T_inf.
"""

from __future__ import annotations

import numpy as np
import fipy as fp


def solve_steady(
    mesh: fp.Grid1D,
    q_inner: float,
    *,
    k: float = 400.0,
    r_outer: float | None = None,
    h: float = 1_000.0,
    T_inf: float = 0.0,
) -> fp.CellVariable:
    """
    Return the analytic steady-state temperature as a FiPy CellVariable.

    Parameters
    ----------
    mesh : fipy.Grid1D
        Radial mesh built by `geometry.build_mesh`.  
        *Faces* must span [r_inner … r_outer].
    q_inner : float
        Axial heat-flux at the inner radius (W m⁻², **positive** into the copper).
    k : float, default 400 W m⁻¹ K⁻¹
        Thermal conductivity of copper.
    r_outer : float | None
        Outer radius.  If *None* it is taken from the rightmost face.
    h : float, default 1000 W m⁻² K⁻¹
        Convective heat-transfer coefficient at the outer surface.
    T_inf : float, default 0 °C
        Ambient temperature.

    Returns
    -------
    fipy.CellVariable
        Temperature field (°C) defined at cell centres.
    """
    r_cells = mesh.cellCenters[0].value
    r_inner = float(mesh.faceCenters[0].value.min())   # leftmost face
    if r_outer is None:
        r_outer = float(mesh.faceCenters[0].value.max())

    # Debugging aid
    print(
        f"[solver] r_inner={r_inner:.4f} m, r_outer={r_outer:.4f} m, "
        f"q_inner={q_inner:.3g} W/m², k={k}"
    )

    # Analytic profile
    T_vals = (  
        (q_inner * r_inner / k) * np.log(r_outer / r_cells)  # conduction term
        + (q_inner * r_inner) / (h * r_outer)                # convective offset
        + T_inf
    )

    # Quick sanity check (can comment out once happy)
    # print(f"[solver] ΔT={T_vals.max() - T_vals.min():.1f} K, "
    #       f"T_outer={T_vals[-1]:.1f} °C")

    return fp.CellVariable(mesh=mesh, name="temperature", value=T_vals)
