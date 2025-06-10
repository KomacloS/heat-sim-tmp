"""Plot helpers for the lumped heatup model."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.typing import NDArray


def plot_heatup(times: NDArray[np.float_], temps: NDArray[np.float_]) -> Figure:
    """Return a Matplotlib Figure showing temperature rise."""
    fig, ax = plt.subplots()
    ax.plot(times, temps)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Pad Temperature")
    ax.xaxis.set_major_formatter(EngFormatter(unit="s"))
    ax.yaxis.set_major_formatter(EngFormatter(unit="°C"))
    return fig


def plot_transient(
    r_centres: NDArray[np.float_], times: NDArray[np.float_], T: NDArray[np.float_]
) -> tuple[Figure, Axes]:
    """Return an animation of radial profiles over time."""

    fig, ax = plt.subplots()
    (line,) = ax.plot(r_centres, T[0])
    ax.set_xlabel("Radius (m)")
    ax.set_ylabel("Temperature (°C)")
    ax.xaxis.set_major_formatter(EngFormatter(unit="m"))
    ax.yaxis.set_major_formatter(EngFormatter(unit="°C"))

    def update(frame: int) -> list[Line2D]:
        line.set_ydata(T[frame])
        ax.set_title(f"t = {times[frame]:.2f} s")
        return [line]

    from matplotlib.animation import FuncAnimation

    FuncAnimation(fig, update, frames=len(times), interval=100)
    return fig, ax


def plot_stack_temperature(
    r_centres: NDArray[np.float_],
    z_centres: NDArray[np.float_],
    T_frame: NDArray[np.float_],
) -> Figure:
    """Return a 2-D temperature colormap for an r-z slice."""

    fig, ax = plt.subplots()
    R, Z = np.meshgrid(r_centres * 1000.0, z_centres * 1000.0)
    pcm = ax.pcolormesh(R, Z, T_frame, shading="auto")
    cbar = fig.colorbar(pcm, ax=ax, label="Temperature (°C)")
    cbar.formatter = EngFormatter(unit="°C")
    cbar.update_ticks()
    ax.set_xlabel("Radius (mm)")
    ax.set_ylabel("z (mm)")
    ax.xaxis.set_major_formatter(EngFormatter(unit="mm"))
    ax.yaxis.set_major_formatter(EngFormatter(unit="mm"))
    return fig
