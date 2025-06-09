import numpy as np
from numpy.typing import NDArray
from laserpad.geometry import build_radial_mesh
from laserpad.solver import solve_transient
from laserpad.beam_profiles import uniform_beam, donut_beam


def test_uniform_flat_profile() -> None:
    r_centres, dr = build_radial_mesh(0.001, 0.002, 20)
    k = 200.0
    rho_cp = 2.0e6
    t_max = 0.05
    dt = 1e-5

    def src(r: NDArray[np.float_]) -> NDArray[np.float_]:
        return uniform_beam(r, 5e4)

    times, T = solve_transient(r_centres, dr, 0.0, k, rho_cp, t_max, dt, src)
    final = T[-1]
    assert np.max(final) - np.min(final) < 0.01 * np.mean(final)


def test_donut_peak_location() -> None:
    r_centres, dr = build_radial_mesh(0.001, 0.003, 30)
    r1 = 0.0015
    r2 = 0.0025
    k = 200.0
    rho_cp = 2.0e6
    t_max = 0.05
    dt = 1e-5

    def src(r: NDArray[np.float_]) -> NDArray[np.float_]:
        return donut_beam(r, r1, r2, 5e4)

    times, T = solve_transient(r_centres, dr, 0.0, k, rho_cp, t_max, dt, src)
    final = T[-1]
    peak_idx = np.argmax(final)
    assert r_centres[peak_idx] >= r1 and r_centres[peak_idx] <= r2


def test_energy_balance() -> None:
    r_centres, dr = build_radial_mesh(0.001, 0.002, 20)
    k = 200.0
    rho_cp = 2.0e6
    t_max = 0.01
    dt = 1e-5
    T0 = 25.0

    def src(r: NDArray[np.float_]) -> NDArray[np.float_]:
        return uniform_beam(r, 5e4)

    times, T = solve_transient(r_centres, dr, 0.0, k, rho_cp, t_max, dt, src, T0)
    final = T[-1]
    energy_in = np.sum(src(r_centres) * 2 * np.pi * r_centres * dr) * t_max
    energy_stored = rho_cp * 2 * np.pi * np.sum((final - T0) * r_centres * dr)
    assert np.isclose(energy_in, energy_stored, rtol=0.01)
