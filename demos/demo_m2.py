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
    power_W = st.number_input("Laser power Qin (W)", value=10.0)
    k = st.number_input("Thermal conductivity k (W/m/K)", value=401.0)
    rho_cp = st.number_input("rho*cp (J/m^3/K)", value=8960.0 * 385.0)
    t_max_ms = st.number_input("Total time (ms)", value=100.0)
    dt_ms = st.number_input("Time step (ms)", value=1.0, format="%.6f")
    max_iter = st.number_input("Max steps", value=1000, min_value=1, step=1)
    allow_unstable = st.checkbox("Ignore stability limit")

    if dt_ms < 0.1:
        st.warning("Time step is very small; simulation may be slow and not optimal.")

    t_max = t_max_ms / 1000.0
    dt = dt_ms / 1000.0

    if "m2_results" not in st.session_state:
        st.session_state["m2_results"] = None

    run = st.button("Run simulation")
    if run:
        r_centres, dr = build_radial_mesh(
            r_inner_mm / 1000.0, r_outer_mm / 1000.0, n_r
        )
        r_inner_m = r_inner_mm / 1000.0
        q_flux = power_W / (2.0 * np.pi * r_inner_m)
        times, T = solve_transient(
            r_centres,
            dr,
            q_flux,
            k,
            rho_cp,
            t_max,
            dt,
            max_steps=int(max_iter),
            allow_unstable=allow_unstable,
        )
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
        r_centres_mm = r_centres * 1000.0
        fig, ax = plt.subplots()
        ax.plot(r_centres_mm, T[t_idx, :])
        ax.set_xlabel("Radius (mm)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"t = {times[t_idx]:.3f} s")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        tt, rr = np.meshgrid(times, r_centres_mm)
        pcm = ax2.pcolormesh(tt, rr, T.T, shading="auto")
        fig2.colorbar(pcm, ax=ax2, label="Temperature (°C)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Radius (mm)")
        ax2.set_title("Temperature vs. time")
        st.pyplot(fig2)

        theta = np.linspace(0.0, 2 * np.pi, 200)
        th, rr2 = np.meshgrid(theta, r_centres_mm)
        temp_ring = np.tile(T[t_idx, :], (len(theta), 1))
        fig3 = plt.figure(figsize=(4, 4))
        ax3 = fig3.add_subplot(111, projection="polar")
        pcm3 = ax3.pcolormesh(th, rr2, temp_ring.T, shading="auto")
        fig3.colorbar(pcm3, ax=ax3, label="Temperature (°C)")
        ax3.set_title("Radial temperature")
        ax3.set_yticklabels([])
        st.pyplot(fig3)


if __name__ == "__main__":
    main()
