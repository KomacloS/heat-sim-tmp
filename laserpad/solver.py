"""Simple lumped-parameter heatup solver."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
