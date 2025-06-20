"""Simple lumped-parameter heatup solver."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable

ProgressCallback = Callable[[int, int], None]


def solve_heatup(
    power_W: float,
    m_kg: float,
    cp: float,
    t_max: float = 1.0,
    dt: float = 0.01,
    T0: float = 25.0,
    *,
    max_steps: int | None = None,
    progress_cb: ProgressCallback | None = None,
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Integrate dT/dt = power/(m*cp) with explicit Euler."""
    times: NDArray[np.float_] = np.arange(0.0, t_max + dt, dt)
    if max_steps is not None:
        times = times[: max_steps + 1]
    temps: NDArray[np.float_] = np.empty_like(times)
    temps[0] = T0
    steps = len(times) - 1
    for i in range(steps):
        temps[i + 1] = temps[i] + (power_W / (m_kg * cp)) * dt
        if progress_cb is not None:
            progress_cb(i + 1, steps)
    return times, temps


def solve_transient(
    r_centres: NDArray[np.float_],
    dr: float,
    q_flux: float,
    k: float,
    rho_cp: float,
    t_max: float,
    dt: float,
    heat_source: Callable[[NDArray[np.float_]], NDArray[np.float_]] | None = None,
    T0: float = 25.0,
    *,
    max_steps: int | None = None,
    allow_unstable: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Explicit transient solver for 1-D cylindrical conduction.

    Parameters
    ----------
    q_flux:
        Applied heat flux at the inner radius [W/m²].
    heat_source:
        Optional callable giving the surface heat-flux distribution ``q''(r)``
        [W/m²]. If ``None`` no volumetric heating is applied. The flux profile
        is converted to a volumetric source ``q''/rho_cp`` in each cell.
    T0:
        Initial temperature.
    max_steps:
        Optional limit on the number of time steps.
    allow_unstable:
        If ``True`` run even when ``dt`` violates the stability limit.
    """

    alpha = k / rho_cp
    dt_lim = 0.5 * dr**2 / alpha
    if dt > dt_lim and not allow_unstable:
        raise ValueError(
            f"Time step {dt:.6f} exceeds stability limit of {dt_lim:.6f} seconds"
        )

    times = np.arange(0.0, t_max + dt, dt)
    if max_steps is not None:
        times = times[: max_steps + 1]
    n_t = len(times)
    steps = n_t - 1
    n_r = len(r_centres)
    T = np.zeros((n_t, n_r), dtype=float)
    T[0, :] = T0

    if heat_source is None:
        q_profile = np.zeros_like(r_centres)
    else:
        q_profile = heat_source(r_centres)
    source = q_profile / rho_cp

    r_faces = np.concatenate(
        [r_centres[:1] - 0.5 * dr, r_centres + 0.5 * dr]
    )  # length n_r + 1

    for n in range(steps):
        old = T[n]
        new = T[n + 1]

        ghost_left = old[0] + dr * q_flux / k
        ghost_right = old[-1]

        T_ext = np.empty(n_r + 2)
        T_ext[0] = ghost_left
        T_ext[1:-1] = old
        T_ext[-1] = ghost_right

        for i in range(n_r):
            r_imh = r_faces[i]
            r_iph = r_faces[i + 1]
            new[i] = old[i] + dt * (
                (alpha / (r_centres[i] * dr**2))
                * (r_iph * (T_ext[i + 2] - old[i]) - r_imh * (old[i] - T_ext[i]))
                + source[i]
            )
        if progress_cb is not None:
            progress_cb(n + 1, steps)

    return times, T


def solve_transient_2d(
    r_centres: NDArray[np.float_],
    dr: float,
    z_centres: NDArray[np.float_],
    dz: float,
    mat_idx: NDArray[np.str_],
    q_flux: float,
    n_t: int,
    dt: float,
    heat_source: Callable[[NDArray[np.float_]], NDArray[np.float_]] | None = None,
    T0: float = 25.0,
    trace_mask: NDArray[np.bool_] | None = None,
    h_trace: float = 1e3,
    T_inf: float = 25.0,
    *,
    max_steps: int | None = None,
    allow_unstable: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Explicit 2-D transient solver in r-z cylindrical coordinates.

    Parameters
    ----------
    trace_mask:
        Boolean array of shape ``(n_theta, n_r)`` describing which angular cells
        are connected to copper traces at the outer radius.
    h_trace:
        Heat-transfer coefficient for trace-connected sectors [W/m²·K].
    T_inf:
        Ambient temperature used for trace heat-sink boundary conditions.
    max_steps:
        Optional limit on the number of time steps.
    allow_unstable:
        If ``True`` run even when ``dt`` violates the stability limit.
    """

    from .geometry import load_materials

    materials = load_materials()

    n_z, n_r = mat_idx.shape

    k = np.zeros((n_z, n_r), dtype=float)
    rho_cp = np.zeros((n_z, n_r), dtype=float)

    for name, props in materials.items():
        mask = mat_idx == name
        k[mask] = props["k"]
        rho_cp[mask] = props["rho"] * props["cp"]

    alpha = k / rho_cp
    dt_lim = 0.55 * min(dr**2, dz**2) / np.max(alpha)
    if dt > dt_lim and not allow_unstable:
        raise ValueError(
            f"Time step {dt:.6f} exceeds stability limit of {dt_lim:.6f} seconds"
        )

    times = np.arange(0.0, (n_t + 1) * dt, dt)
    if max_steps is not None:
        times = times[: max_steps + 1]
    steps = len(times) - 1
    T = np.full((steps + 1, n_z, n_r), T0, dtype=float)

    if heat_source is None:
        q_profile = np.zeros_like(r_centres)
    else:
        q_profile = heat_source(r_centres)
    source_r = q_profile / rho_cp[0, :]

    r_faces = np.concatenate([r_centres[:1] - 0.5 * dr, r_centres + 0.5 * dr])

    if trace_mask is not None:
        frac_trace = np.mean(trace_mask, axis=0)
    else:
        frac_trace = np.zeros_like(r_centres)

    for n in range(steps):
        old = T[n]
        new = T[n + 1]

        ghost_r_left = old[:, 0] + dr * q_flux / k[:, 0]

        h_eff = frac_trace[-1] * h_trace
        ghost_r_right = old[:, -1] - dr * h_eff / k[:, -1] * (old[:, -1] - T_inf)

        T_r = np.empty((n_z, n_r + 2))
        T_r[:, 0] = ghost_r_left
        T_r[:, 1:-1] = old
        T_r[:, -1] = ghost_r_right

        T_z = np.empty((n_z + 2, n_r))
        T_z[0, :] = old[0, :]
        T_z[1:-1, :] = old
        T_z[-1, :] = old[-1, :]

        for j in range(n_z):
            for i in range(n_r):
                r_imh = r_faces[i]
                r_iph = r_faces[i + 1]
                radial = (
                    alpha[j, i]
                    / (r_centres[i] * dr**2)
                    * (
                        r_iph * (T_r[j, i + 2] - old[j, i])
                        - r_imh * (old[j, i] - T_r[j, i])
                    )
                )
                axial = (
                    alpha[j, i] * (T_z[j + 2, i] - 2.0 * old[j, i] + T_z[j, i]) / dz**2
                )
                new[j, i] = old[j, i] + dt * (radial + axial + source_r[i])
        if progress_cb is not None:
            progress_cb(n + 1, steps)

    return times, T
