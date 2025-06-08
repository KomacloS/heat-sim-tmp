# tests/test_m1.py

import numpy as np
import pytest

from laserpad.geometry import build_mesh
from laserpad.solver import solve_steady


@pytest.fixture(scope="module")
def mesh_and_temp() -> tuple[np.ndarray, np.ndarray, float]:
    # Default parameters from spec:
    r_inner = 0.50
    r_outer = 1.50
    q_flux = 1.0e6
    n_r = 100

    r_centres, dr = build_mesh(r_inner=r_inner, r_outer=r_outer, n_r=n_r)
    temperature = solve_steady(
        r_centres=r_centres, dr=dr, q_inner=q_flux, k=400.0, r_outer=r_outer
    )

    return r_centres, temperature, dr[0]


def test_delta_T_large(mesh_and_temp: tuple[np.ndarray, np.ndarray, float]) -> None:
    """Test 1: Ensure that max ΔT across the ring is > 10 K."""
    r_centres, temperature, _ = mesh_and_temp
    T_vals = temperature.copy()
    delta_T = np.max(T_vals) - np.min(T_vals)
    assert delta_T > 10.0, f"ΔT too small: {delta_T:.4f} K; expected > 10 K."


def test_monotonic_decrease(mesh_and_temp: tuple[np.ndarray, np.ndarray, float]) -> None:
    """Test 2: Ensure T(r) is monotonically decreasing with r."""
    r_cell, T_vals, _ = mesh_and_temp
    r_cell = r_cell.copy()
    T_vals = T_vals.copy()

    # Sort by radius just in case (though mesh is already ordered)
    idx_sort = np.argsort(r_cell)
    T_sorted = T_vals[idx_sort]
    # Check that each subsequent T is <= previous T (monotonic non-increasing)
    diffs = np.diff(T_sorted)
    assert np.all(
        diffs <= 1e-8
    ), "Temperature is not monotonically decreasing with radius."


def test_energy_balance_flux(mesh_and_temp: tuple[np.ndarray, np.ndarray, float]) -> None:
    """Inner heat-flux ≈ outer convective heat-loss (< 1 % mismatch)."""
    import math

    r_cell, temperature, dr = mesh_and_temp

    r_inner = float(r_cell.min() - dr / 2)
    r_outer = float(r_cell.max() + dr / 2)

    q_inner = 1.0e6
    h = 1_000.0
    T_inf = 0.0

    q_in = q_inner * 2 * math.pi * r_inner

    T_vals = temperature
    r_vals = r_cell
    T_outer_face = T_vals[-1] + (T_vals[-1] - T_vals[-2]) / (
        r_vals[-1] - r_vals[-2]
    ) * (r_outer - r_vals[-1])
    q_out = -h * (T_outer_face - T_inf) * 2 * math.pi * r_outer

    imbalance = abs(q_in + q_out) / abs(q_in)
    assert imbalance < 0.01, f"Energy imbalance {imbalance*100:.2f}% > 1 %"
