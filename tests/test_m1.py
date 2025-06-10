from laserpad.geometry import get_pad_properties
from laserpad.solver import solve_heatup
import numpy as np


def test_heatup_positive_delta() -> None:
    props = get_pad_properties(1.0)
    times, temps = solve_heatup(
        1000.0, props["mass_kg"], props["heat_capacity_J_per_K"], 0.1, 0.02, 25.0
    )
    assert temps[-1] > temps[0]


def test_monotonic_increase() -> None:
    props = get_pad_properties(1.0)
    _, temps = solve_heatup(
        1000.0, props["mass_kg"], props["heat_capacity_J_per_K"], 0.1, 0.02, 25.0
    )
    assert np.all(np.diff(temps) >= 0), "Temperature must never decrease"


def test_energy_balance() -> None:
    props = get_pad_properties(1.0)
    P = 1000.0
    m = props["mass_kg"]
    cp = props["heat_capacity_J_per_K"]
    T0 = 25.0
    times, temps = solve_heatup(P, m, cp, 0.1, 0.02, T0)
    t_max = times[-1]
    expected = T0 + P * t_max / (m * cp)
    assert abs(temps[-1] - expected) < 0.01 * abs(
        expected
    ), f"Energy mismatch: got {temps[-1]:.3f}, expected ~{expected:.3f}"
