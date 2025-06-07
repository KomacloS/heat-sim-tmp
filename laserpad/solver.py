"""laserpad.solver – steady-state cylindrical conduction

Internal units: **metres**.

BCs
----
• Inner rim  (r = r_inner) : -k ∂T/∂r  =  q_inner        (Neumann heat-flux)
• Outer rim  (r = r_outer) : -k ∂T/∂r  =  h (T – T_inf)  (Robin convection)

Default material & convection data
----------------------------------
k      = 400.0  # W m⁻¹ K⁻¹  (copper)
h      = 1000.0 # W m⁻² K⁻¹
T_inf  =   0.0  # °C reference (can be 0 K because only differences matter)
"""
from typing import Optional

import fipy as fp


def solve_steady(
    mesh: fp.Grid1D,
    q_inner: float,                # W m⁻², applied at r_inner
    k: float = 400.0,
    r_outer: Optional[float] = None,
    h: float = 1_000.0,
    T_inf: float = 0.0,
) -> fp.CellVariable:
    """Return FiPy `CellVariable` with temperature in **kelvin**."""
    # ───────── mesh radii in SI (metres) ─────────
    dr = float(mesh.dx)                               # dx already in m
    r_cell = mesh.cellCenters[0].value.copy()         # absolute radii [m]
    r_inner = float(r_cell.min() - dr / 2)            # inner face radius
    if r_outer is None:
        r_outer = float(r_cell.max() + dr / 2)        # outer face radius

    # ───────── variable initialised to T_inf ─────
    T = fp.CellVariable(mesh=mesh, value=T_inf, name="temperature")

    # ───────── inner Neumann (heat-flux) ─────────
    #   -k ∂T/∂r = q_inner  ⇒  ∂T/∂r = -q_inner / k
    T.faceGrad.constrain((-q_inner / k,), where=mesh.facesLeft)

    # ───────── governing equation (axisymmetric) ─
    eq = fp.DiffusionTerm(coeff=k)

    # ───────── outer Robin convection ────────────
    # We apply   h·(T–T_inf)  by splitting into
    #   ImplicitSourceTerm: +h·T
    #   Explicit Source   : −h·T_inf
    beta = fp.CellVariable(mesh=mesh, value=0.0)
    beta[-1] = 1.0                                  # mask last cell only
    eq += fp.ImplicitSourceTerm(coeff=(h / k) * beta, var=T)
    eq += (h * T_inf / k) * beta                    # constant source

    # ───────── solve ─────────────────────────────
    eq.solve(var=T)
    return T
