"""Plot helpers for the lumped heatup model."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray


def plot_heatup(times: NDArray[np.float_], temps: NDArray[np.float_]) -> Figure:
    """Return a Matplotlib Figure showing temperature rise."""
    fig, ax = plt.subplots()
    ax.plot(times, temps)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Pad Temperature")
    return fig


def plot_transient(
    r_centres: NDArray[np.float_],
    times: NDArray[np.float_],
    T: NDArray[np.float_],
) -> Figure:
    """Return an animation of radial profiles over time."""
    fig, ax = plt.subplots()
    line, = ax.plot(r_centres, T[0])
    ax.set_xlabel("Radius (m)")
    ax.set_ylabel("Temperature (°C)")

    def update(frame: int) -> tuple[list[plt.Line2D]]:
        line.set_ydata(T[frame])
        ax.set_title(f"t = {times[frame]:.2f} s")
        return [line]

    from matplotlib.animation import FuncAnimation

    FuncAnimation(fig, update, frames=len(times), interval=100)
    return fig
