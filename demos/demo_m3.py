"""Beam-shape transient demo for Milestone 3."""

from __future__ import annotations

import streamlit as st
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from laserpad.geometry import build_radial_mesh
from laserpad.solver import solve_transient
from laserpad.beam_profiles import uniform_beam, gaussian_beam, donut_beam


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
    t_max_ms = st.number_input("Total time (ms)", value=100.0)
    dt_ms = st.number_input("Time step (ms)", value=0.1)
    max_iter = int(
        st.number_input("Maximum iterations", value=1000, min_value=1, step=100)
    )

    if dt_ms < 0.01:
        st.warning("Time step is very small; simulation may be slow and not optimal.")

    t_max = t_max_ms / 1000.0
    dt = dt_ms / 1000.0

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

        alpha = k / rho_cp
        dt_lim = 0.5 * dr**2 / alpha
        if dt > dt_lim:
            st.warning(
                f"Time step {dt:.6f} s exceeds stability limit {dt_lim:.6f} s"
            )
        import warnings

        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            times, T = solve_transient(
                r_centres,
                dr,
                0.0,
                k,
                rho_cp,
                t_max,
                dt,
                src,
                max_steps=max_iter,
            )
        for w in warns:
            st.warning(str(w.message))

        t_idx = st.slider("Time index", 0, len(times) - 1, 0)
        r_centres_mm = r_centres * 1000.0
        fig, ax = plt.subplots()
        ax.plot(r_centres_mm, T[t_idx, :])
        ax.set_xlabel("Radius (mm)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"Beam: {beam_type}, t = {times[t_idx]:.6f} s")
        st.pyplot(fig)

        theta = np.linspace(0.0, 2 * np.pi, 200)
        th, rr = np.meshgrid(theta, r_centres_mm)
        temp_ring = np.tile(T[t_idx, :], (len(theta), 1))
        fig2 = plt.figure(figsize=(4, 4))
        ax2 = fig2.add_subplot(111, projection="polar")
        pcm = ax2.pcolormesh(th, rr, temp_ring.T, shading="auto")
        fig2.colorbar(pcm, ax=ax2, label="Temperature (°C)")
        ax2.set_title("Radial temperature")
        ax2.set_yticklabels([])
        st.pyplot(fig2)


if __name__ == "__main__":
    main()
