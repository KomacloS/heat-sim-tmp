import numpy as np

from laserpad.geometry import build_stack_mesh, load_materials
from laserpad.solver import solve_transient_2d


def test_energy_conservation() -> None:
    r_centres, dr, z_centres, dz, mat_idx = build_stack_mesh(
        0.001, 0.003, 10, 0.000035, 0.0002, 5
    )
    q_flux = 1e5
    n_t = 10
    dt = 5e-6
    times, T = solve_transient_2d(
        r_centres, dr, z_centres, dz, mat_idx, q_flux, n_t, dt
    )
    t_max = times[-1]

    mats = load_materials()
    rho_cp = np.zeros_like(mat_idx, dtype=float)
    for name, props in mats.items():
        mask = mat_idx == name
        rho_cp[mask] = props["rho"] * props["cp"]

    final = T[-1]
    volumes = 2 * np.pi * dr * dz * np.outer(np.ones_like(z_centres), r_centres)
    energy_stored = np.sum(rho_cp * volumes * (final - 25.0))

    r_inner = r_centres[0] - dr / 2
    height = z_centres[-1] + dz / 2
    energy_in = q_flux * 2 * np.pi * r_inner * height * t_max
    assert np.isclose(energy_in, energy_stored, rtol=0.1)


def test_material_interface_drop() -> None:
    r_centres, dr, z_centres, dz, mat_idx = build_stack_mesh(
        0.001, 0.002, 8, 0.000035, 0.0002, 6
    )
    times, T = solve_transient_2d(r_centres, dr, z_centres, dz, mat_idx, 5e4, 5, 5e-6)
    final = T[-1]

    pad_idx = np.where(z_centres < 0.000035)[0][-1]
    sub_idx = np.where(z_centres >= 0.000035)[0][0]
    assert final[sub_idx, 0] - final[pad_idx, 0] >= 0


def test_cfl_enforced() -> None:
    r_centres, dr, z_centres, dz, mat_idx = build_stack_mesh(
        0.001, 0.002, 10, 0.000035, 0.0002, 5
    )
    mats = load_materials()
    alpha_max = max(
        props["k"] / (props["rho"] * props["cp"]) for props in mats.values()
    )
    dt_bad = 0.6 * min(dr**2, dz**2) / alpha_max
    try:
        solve_transient_2d(r_centres, dr, z_centres, dz, mat_idx, 1e5, 1, dt_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unstable dt")
