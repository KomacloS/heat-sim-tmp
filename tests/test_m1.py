# tests/test_m1.py

import numpy as np
import pytest
import fipy as fp

from laserpad.geometry import build_mesh
from laserpad.solver import solve_steady


@pytest.fixture(scope="module")
def mesh_and_temp():
    # Default parameters from spec:
    r_inner = 0.50
    r_outer = 1.50
    q_flux = 1.0e6
    n_r = 100

    mesh = build_mesh(r_inner=r_inner, r_outer=r_outer, n_r=n_r)
    temperature = solve_steady(mesh=mesh, q_inner=q_flux, k=400.0, r_outer=r_outer)

    return mesh, temperature


def test_delta_T_large(mesh_and_temp):
    """Test 1: Ensure that max ΔT across the ring is > 10 K."""
    mesh, temperature = mesh_and_temp
    T_vals = temperature.value.copy()
    delta_T = np.max(T_vals) - np.min(T_vals)
    assert delta_T > 10.0, f"ΔT too small: {delta_T:.4f} K; expected > 10 K."


def test_monotonic_decrease(mesh_and_temp):
    """Test 2: Ensure T(r) is monotonically decreasing with r."""
    mesh, temperature = mesh_and_temp
    r_cell = mesh.cellCenters[0].value.copy()
    T_vals = temperature.value.copy()

    # Sort by radius just in case (though mesh is already ordered)
    idx_sort = np.argsort(r_cell)
    T_sorted = T_vals[idx_sort]
    # Check that each subsequent T is <= previous T (monotonic non-increasing)
    diffs = np.diff(T_sorted)
    assert np.all(diffs <= 1e-8), "Temperature is not monotonically decreasing with radius."


def test_energy_balance_flux(mesh_and_temp):
    """Test 3: Inner flux ≫ outer flux for an insulated outer boundary."""
    mesh, temperature = mesh_and_temp
    r_cell = mesh.cellCenters[0].value.copy()
    dr = mesh.dx
    q_inner = 1.0e6
    k = 400.0

    # Compute inner heat–flux via finite difference
    T = temperature.value.copy()
    dT_dr_inner = (T[1] - T[0]) / dr
    flux_in = -k * dT_dr_inner

    # Compute outer flux via backward difference
    dT_dr_outer = (T[-1] - T[-2]) / dr
    flux_out = -k * dT_dr_outer

    # Outer flux should be near zero for insulated BC
    assert abs(flux_out) < 0.01 * abs(flux_in), \
        f"Outer flux {flux_out:.3e} not ≪ inner flux {flux_in:.3e}"

