import math
from typing import Any, Tuple

import numpy as np
import pytest

from laserpad.geometry import build_mesh
from laserpad.solver import solve_steady


@pytest.fixture(scope="module")
def mesh_and_temp() -> Tuple[float, float, float, Any]:
    r_inner = 0.50
    r_outer = 1.50
    q_inner = 1.0e6
    n_cells = 300

    mesh = build_mesh(r_inner=r_inner, r_outer=r_outer, n_r=n_cells)
    temperature = solve_steady(
        mesh=mesh,
        q_inner=q_inner,
        k=400.0,
        r_outer=r_outer,
        h=1_000.0,
        T_inf=0.0,
        r_inner=r_inner,
    )
    return r_inner, r_outer, q_inner, temperature


def test_delta_T_large(mesh_and_temp: Tuple[float, float, float, Any]) -> None:
    _, _, _, temperature = mesh_and_temp
    T_vals = np.array(temperature.value)
    assert T_vals[0] - T_vals[-1] > 10.0


def test_monotonic_decrease(mesh_and_temp: Tuple[float, float, float, Any]) -> None:
    _, _, _, temperature = mesh_and_temp
    T_vals = np.array(temperature.value)
    assert np.all(np.diff(T_vals) <= 0)


def test_energy_balance(mesh_and_temp: Tuple[float, float, float, Any]) -> None:
    r_inner, r_outer, q_inner, temperature = mesh_and_temp
    h = 1_000.0
    T_inf = 0.0
    T_outer = float(temperature.value[-1])

    q_in = q_inner * 2 * math.pi * r_inner
    q_out = -h * 2 * math.pi * r_outer * (T_outer - T_inf)
    assert abs(q_in + q_out) < 0.01 * abs(q_in)
