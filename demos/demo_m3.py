"""Beam-shape transient demo for Milestone 3."""

from __future__ import annotations

import streamlit as st
import numpy as np
from numpy.typing import NDArray

from laserpad.geometry import build_radial_mesh
from laserpad.solver import solve_transient
from laserpad.beam_profiles import uniform_beam, gaussian_beam, donut_beam
from laserpad.plot import plot_transient


def main() -> None:
    st.title("M3: Beam-Shape Transient Heat-Up")

    r_in = st.number_input("r_inner (mm)", value=0.5) / 1000.0
    r_out = st.number_input("r_outer (mm)", value=1.5) / 1000.0
    n_r = st.slider("Radial cells", 20, 200, 50)
    beam_type = st.selectbox("Beam type", ["Uniform", "Gaussian", "Donut"])
    q0 = st.number_input("Peak flux (W/m²)", value=1e6)

    if beam_type == "Gaussian":
        sigma = st.number_input("Gaussian σ (mm)", value=0.5) / 1000.0
    if beam_type == "Donut":
        r1 = st.number_input("Inner radius (mm)", value=0.5) / 1000.0
        r2 = st.number_input("Outer radius (mm)", value=1.0) / 1000.0

    k = st.number_input("k (W/m·K)", value=400.0)
    rho_cp = st.number_input("ρ·cₚ (J/m³·K)", value=8.96e6)
    t_max = st.number_input("Total time (s)", value=0.1)
    dt = st.number_input("Time step (s)", value=1e-4)

    if st.button("Run"):
        r_centres, dr = build_radial_mesh(r_in, r_out, n_r)

        if beam_type == "Uniform":

            def src(r: NDArray[np.float_]) -> NDArray[np.float_]:
                return uniform_beam(r, q0)

        elif beam_type == "Gaussian":

            def src(r: NDArray[np.float_]) -> NDArray[np.float_]:
                return gaussian_beam(r, q0, sigma)

        else:

            def src(r: NDArray[np.float_]) -> NDArray[np.float_]:
                return donut_beam(r, r1, r2, q0)

        times, T = solve_transient(r_centres, dr, 0.0, k, rho_cp, t_max, dt, src)

        t_idx = st.slider("Time index", 0, len(times) - 1, 0)
        fig, ax = plot_transient(r_centres, times, T)
        ax.set_title(f"Beam: {beam_type}, t={times[t_idx]:.3f}s")
        st.pyplot(fig)


if __name__ == "__main__":
    main()
