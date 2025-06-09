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
