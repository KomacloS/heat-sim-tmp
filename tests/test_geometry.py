import math
from laserpad.geometry import get_annular_pad_properties


def test_annular_mass_and_heat_capacity() -> None:
    r_in_mm, r_out_mm = 0.5, 1.5
    t_mm = 0.035
    density = 8000.0
    cp = 250.0
    props = get_annular_pad_properties(r_in_mm, r_out_mm, t_mm, density, cp)
    r_in = r_in_mm * 1e-3
    r_out = r_out_mm * 1e-3
    t = t_mm * 1e-3
    area = math.pi * (r_out**2 - r_in**2)
    volume = area * t
    mass = density * volume
    heat_capacity = mass * cp
    assert math.isclose(props["area_m2"], area, rel_tol=1e-9)
    assert math.isclose(props["volume_m3"], volume, rel_tol=1e-9)
    assert math.isclose(props["mass_kg"], mass, rel_tol=1e-9)
    assert math.isclose(props["heat_capacity_J_per_K"], heat_capacity, rel_tol=1e-9)
