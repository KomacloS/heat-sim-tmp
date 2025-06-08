# laserpad/plot.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_ring(r_centres: np.ndarray, T: np.ndarray) -> Figure:
    """Create a polar-colored 2D plot of temperature on an annular ring.

    The function spins the 1‑D radial temperature profile ``T`` about the
    origin to create a 2‑D polar ``pcolormesh``.

    Args:
        r_centres: Radial cell-centre positions.
        T: Temperature values at ``r_centres``.

    Returns:
        A Matplotlib Figure with a polar pcolormesh, and a colorbar labelled
        ``"°C"``.
    """
    n_r = len(r_centres)

    # Estimate cell edges assuming near-uniform spacing
    edges = np.empty(n_r + 1)
    edges[1:-1] = (r_centres[:-1] + r_centres[1:]) / 2.0
    dr0 = r_centres[1] - r_centres[0]
    edges[0] = r_centres[0] - dr0 / 2.0
    dr_last = r_centres[-1] - r_centres[-2]
    edges[-1] = r_centres[-1] + dr_last / 2.0

    r_inner = edges[0]
    r_outer = edges[-1]

    # Number of angular divisions for smooth ring:
    n_theta = 200
    theta_edges = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)

    # Build a meshgrid in (θ, r) for the boundaries:
    Theta, R = np.meshgrid(theta_edges, edges)

    # Build the 2D Z-array: each ring cell has a constant T from the 1D solution
    Z = np.repeat(T[:, np.newaxis], n_theta, axis=1)

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
