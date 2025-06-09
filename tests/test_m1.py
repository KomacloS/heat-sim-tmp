from laserpad.geometry import get_pad_properties
from laserpad.solver import solve_heatup


def test_heatup_positive_delta() -> None:
    props = get_pad_properties(1.0)
    times, temps = solve_heatup(
        1000.0, props["mass_kg"], props["heat_capacity_J_per_K"], 0.1, 0.02, 25.0
    )
    assert temps[-1] > temps[0]
