"""Simple lumped-parameter heatup solver."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable


def solve_heatup(
    power_W: float,
    m_kg: float,
    cp: float,
    t_max: float = 1.0,
    dt: float = 0.01,
    T0: float = 25.0,
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Integrate dT/dt = power/(m*cp) with explicit Euler."""
    times: NDArray[np.float_] = np.arange(0.0, t_max + dt, dt)
    temps: NDArray[np.float_] = np.empty_like(times)
    temps[0] = T0
    for i in range(len(times) - 1):
        temps[i + 1] = temps[i] + (power_W / (m_kg * cp)) * dt
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
    """

    alpha = k / rho_cp
    dt_lim = 0.5 * dr**2 / alpha
    if dt > dt_lim:
        raise ValueError(
            f"Time step {dt:g} exceeds stability limit of {dt_lim:g} seconds"
        )

    times = np.arange(0.0, t_max + dt, dt)
    n_t = len(times)
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

    for n in range(n_t - 1):
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

    return times, T
