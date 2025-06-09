"""Spatially resolved transient demo for Milestone 2."""

from __future__ import annotations

import streamlit as st
import numpy as np

from laserpad.geometry import build_radial_mesh
from laserpad.solver import solve_transient
from laserpad.plot import plot_transient


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

    if st.button("Run simulation"):
        r_centres, dr = build_radial_mesh(r_inner_mm / 1000.0, r_outer_mm / 1000.0, n_r)
        times, T = solve_transient(r_centres, dr, q_flux, k, rho_cp, t_max, dt)
        fig = plot_transient(r_centres, times, T)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
