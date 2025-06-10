import json
import numpy as np
from laserpad.geometry import build_stack_mesh_with_traces, load_materials
from laserpad.solver import solve_transient_2d


TRACE_HALF = json.dumps([{"start_angle": 0, "end_angle": 180}])
TRACE_FULL = json.dumps([{"start_angle": 0, "end_angle": 360}])


def run_case(trace_json: str, h: float = 1e3) -> tuple[float, float, float]:
    traces = json.loads(trace_json)
    r, dr, z, dz, mat_idx, mask = build_stack_mesh_with_traces(
        0.001,
        0.003,
        10,
        0.000035,
        0.0002,
        5,
        [(t["start_angle"], t["end_angle"]) for t in traces],
    )
    n_t = 200
    dt = 1e-5
    q_flux = 1e6
    times, T = solve_transient_2d(
        r,
        dr,
        z,
        dz,
        mat_idx,
        q_flux,
        n_t,
        dt,
        trace_mask=mask,
        h_trace=h,
        T_inf=25.0,
    )
    t_max = times[-1]
    mats = load_materials()
    rho_cp = np.zeros_like(mat_idx, dtype=float)
    for name, props in mats.items():
        mask_mat = mat_idx == name
        rho_cp[mask_mat] = props["rho"] * props["cp"]
    final = T[-1]
    volumes = 2 * np.pi * dr * dz * np.outer(np.ones_like(z), r)
    energy_stored = np.sum(rho_cp * volumes * (final - 25.0))
    r_inner = r[0] - dr / 2
    height = z[-1] + dz / 2
    energy_in = q_flux * 2 * np.pi * r_inner * height * t_max
    frac = np.mean(mask, axis=0)[-1]
    r_outer = r[-1] + dr / 2
    energy_loss = 0.0
    for n in range(len(times) - 1):
        boundary = T[n, :, -1]
        energy_loss += (
            np.sum(frac * h * (boundary - 25.0) * 2 * np.pi * r_outer * dz) * dt
        )
    return energy_in, energy_stored, energy_loss


def test_half_vs_full_flux_ratio() -> None:
    ein1, stored1, loss1 = run_case(TRACE_HALF)
    ein2, stored2, loss2 = run_case(TRACE_FULL)
    assert np.isclose(loss1 / loss2, 0.5, rtol=0.1)


def test_energy_balance_with_traces() -> None:
    ein, stored, loss = run_case(TRACE_HALF)
    assert np.isclose(ein, stored + loss, rtol=0.1)


def test_no_traces_is_adiabatic() -> None:
    empty = json.dumps([])
    ein, stored, loss = run_case(empty)
    assert np.isclose(ein, stored, rtol=0.1)
    assert loss < 1e-6
