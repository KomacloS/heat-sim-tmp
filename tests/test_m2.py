import numpy as np
from laserpad.geometry import build_radial_mesh
from laserpad.solver import solve_transient


def test_energy_conservation() -> None:
    r_centres, dr = build_radial_mesh(0.001, 0.002, 20)
    q = 5e4
    k = 200.0
    rho_cp = 2.0e6
    t_max = 0.01
    dt = 1e-5
    times, T = solve_transient(r_centres, dr, q, k, rho_cp, t_max, dt)
    final = T[-1]
    r_inner = r_centres[0] - dr / 2
    energy_in = q * 2 * np.pi * r_inner * t_max
    energy_stored = rho_cp * 2 * np.pi * np.sum((final - 25.0) * r_centres * dr)
    assert np.isclose(energy_in, energy_stored, rtol=0.01)


def test_near_uniform_long_time() -> None:
    r_centres, dr = build_radial_mesh(0.001, 0.002, 20)
    q = 5e4
    k = 200.0
    rho_cp = 2.0e6
    alpha = k / rho_cp
    t_max = 0.1
    dt = 1e-5
    times, T = solve_transient(r_centres, dr, q, k, rho_cp, t_max, dt)
    final = T[-1]
    assert np.max(final) - np.min(final) < 0.01 * np.mean(final)


def test_cfl_violation() -> None:
    r_centres, dr = build_radial_mesh(0.001, 0.002, 20)
    q = 1e4
    k = 200.0
    rho_cp = 2.0e6
    alpha = k / rho_cp
    dt_bad = 0.6 * dr**2 / alpha  # violates 0.5 dr^2/alpha
    try:
        solve_transient(r_centres, dr, q, k, rho_cp, 0.01, dt_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unstable dt")
