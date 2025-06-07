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


# tests/test_m1.py (only the energy‐balance test changes)
def test_energy_balance_flux(mesh_and_temp):
    import numpy as np
    mesh, T = mesh_and_temp
    q_inner = 1e6
    h = 1e3
    T_inf = 0.0
    k = 400.0

    # Compute flux_in by finite‐difference at first two cells:
    dr = mesh.dx
    Tvals = T.value
    dTdr_in = (Tvals[1] - Tvals[0]) / dr
    flux_in = -k * dTdr_in

    # Compute flux_out similarly at last two cells:
    dTdr_out = (Tvals[-1] - Tvals[-2]) / dr
    # That is dT/dr at r_outer; convection flux = -k dT/dr
    flux_conv = -k * dTdr_out

    # It should match h*(T_outer - T_inf)
    # Assert that |flux_in - flux_conv| is small:
    assert abs(flux_in - flux_conv) < 0.01 * abs(flux_in)
