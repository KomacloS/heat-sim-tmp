# laserpad/plot.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_ring(
    r_centres: np.ndarray, temperature: np.ndarray, r_inner: float, r_outer: float
) -> Figure:
    """Create a polar-colored 2D plot of temperature on an annular ring.

    We take the 1D radial solution (mesh + temperature) and spin it around θ ∈ [0, 2π]
    to produce a “donut-shaped” pcolormesh.

    Args:
        r_centres: Radial cell centres.
        temperature: Temperature array matching ``r_centres``.
        r_inner: Inner radius of the ring.
        r_outer: Outer radius of the ring.

    Returns:
        A Matplotlib Figure with a polar pcolormesh, and a colorbar labeled “°C”.
    """
    n_r = len(r_centres)

    # Build radial edges (n_r + 1):
    r_edges = np.linspace(r_inner, r_outer, n_r + 1)

    # Number of angular divisions for smooth ring:
    n_theta = 200
    theta_edges = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)

    # Build a meshgrid in (θ, r) for the boundaries:
    Theta, R = np.meshgrid(theta_edges, r_edges)

    # Build the 2D Z-array: each ring cell has a constant T from the 1D solution:
    # temperature.value has length n_r. We want shape (n_r, n_theta), so broadcast:
    Z = np.zeros((n_r, n_theta))
    for i in range(n_r):
        Z[i, :] = float(temperature[i])

    # Create polar plot:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    pcm = ax.pcolormesh(Theta, R, Z, shading="auto")
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label("°C")
    ax.set_title("Steady-State Temperature on Annular Copper Pad")
    ax.set_ylim((r_inner, r_outer))
    ax.set_yticks([])
    ax.set_xticks([])

    return fig
