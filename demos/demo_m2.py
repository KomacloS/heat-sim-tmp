"""Spatially resolved transient demo for Milestone 2."""

from __future__ import annotations

import streamlit as st
import matplotlib.pyplot as plt

import numpy as np

from laserpad.geometry import build_radial_mesh
from laserpad.solver import solve_transient


def main() -> None:
    st.title("M2: Transient Radial Heatup Demo")

    r_inner_mm = st.number_input("Inner radius (mm)", value=0.25, min_value=0.01)
    r_outer_mm = st.number_input("Outer radius (mm)", value=0.5, min_value=0.1)
    n_r = st.slider("Radial cells", min_value=5, max_value=100, value=20)
    q_flux = st.number_input("Heat flux at inner radius (W/m^2)", value=1e4)
    k = st.number_input("Thermal conductivity k (W/m/K)", value=401.0)
    rho_cp = st.number_input("rho*cp (J/m^3/K)", value=8960.0 * 385.0)
    t_max = st.number_input("Total time (s)", value=0.1)
    dt = st.number_input("Time step (s)", value=0.001)

    if "m2_results" not in st.session_state:
        st.session_state["m2_results"] = None

    run = st.button("Run simulation")
    if run:
        r_centres, dr = build_radial_mesh(
            r_inner_mm / 1000.0, r_outer_mm / 1000.0, n_r
        )
        times, T = solve_transient(r_centres, dr, q_flux, k, rho_cp, t_max, dt)
        st.session_state["m2_results"] = (r_centres, times, T)

    if st.session_state["m2_results"] is not None:
        r_centres, times, T = st.session_state["m2_results"]

        t_idx = st.slider(
            "Time step",
            min_value=0,
            max_value=len(times) - 1,
            value=0,
            step=1,
        )
        fig, ax = plt.subplots()
        ax.plot(r_centres, T[t_idx, :])
        ax.set_xlabel("Radius (m)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"t = {times[t_idx]:.3f} s")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        tt, rr = np.meshgrid(times, r_centres)
        pcm = ax2.pcolormesh(tt, rr, T.T, shading="auto")
        fig2.colorbar(pcm, ax=ax2, label="Temperature (°C)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Radius (m)")
        ax2.set_title("Temperature vs. time")
        st.pyplot(fig2)


if __name__ == "__main__":
    main()
