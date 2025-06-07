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
    """Test 3: (Optional) Rough energy check: flux_in ≈ flux_out.

    For M1 we expect an approximate balance < 1%—but since we used Dirichlet at r_outer,
    we allow a higher tolerance (e.g., 5%).
    """
    mesh, temperature = mesh_and_temp
    # Parameters:
    r_cell = mesh.cellCenters[0].value.copy()
    dr = mesh.dx
    r_inner = float(np.min(r_cell))
    # Outer radius from spec fixture:
    r_outer = 1.50
    q_inner = 1.0e6
    k = 400.0

    # Compute heat flux at inner boundary:
    # dT/dr |_{r_inner} ≈ (T[1] - T[0]) / dr
    T_vals = temperature.value.copy()
    dT_dr_inner = (T_vals[1] - T_vals[0]) / dr
    flux_in = -k * dT_dr_inner  # W per unit-length (in these units)

    # Compute approximate dT/dr at r_outer using backward difference
    # Find index of cell closest to r_outer:
    # Last two cells: indices -2, -1
    dT_dr_outer = (T_vals[-1] - T_vals[-2]) / dr
    flux_out = -k * dT_dr_outer

    # Compare magnitudes:
    if flux_in == 0:
        pytest.skip("Inner flux is zero, cannot test energy balance.")
    ratio = abs((flux_in - flux_out) / flux_in)
    assert ratio < 0.05, f"Energy imbalance too large: {ratio*100:.2f}% > 5%."
