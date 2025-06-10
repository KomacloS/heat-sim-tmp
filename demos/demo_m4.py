"""Multilayer stack-up transient demo for Milestone 4."""

from __future__ import annotations

import streamlit as st

from laserpad.geometry import build_stack_mesh
from laserpad.solver import solve_transient_2d
from laserpad.plot import plot_stack_temperature


def main() -> None:
    st.title("M4: Multilayer Stack-Up")

    r_in = st.number_input("r_inner (mm)", value=0.5) / 1000.0
    r_out = st.number_input("r_outer (mm)", value=1.5) / 1000.0
    n_r = st.slider("Radial cells", 10, 200, 50)
    pad_th = st.number_input("Pad thickness (mm)", value=0.035) / 1000.0
    sub_th = st.number_input("Substrate thickness (mm)", value=0.2) / 1000.0
    n_z = st.slider("Axial cells", 10, 200, 50)
    power_W = st.number_input("Laser power Qin (W)", value=10.0)
    n_t = st.slider("Time steps", 10, 200, 50)
    dt_ms = st.number_input("Time step (ms)", value=0.1)

    if dt_ms < 0.01:
        st.warning("Time step is very small; simulation may be slow and not optimal.")

    dt = dt_ms / 1000.0

    if st.button("Run"):
        r_centres, dr, z_centres, dz, mat_idx = build_stack_mesh(
            r_in, r_out, n_r, pad_th, sub_th, n_z
        )
        height = pad_th + sub_th
        q_flux = power_W / (2.0 * np.pi * r_in * height)
        import warnings

        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            times, T = solve_transient_2d(
                r_centres, dr, z_centres, dz, mat_idx, q_flux, n_t, dt
            )
        for w in warns:
            st.warning(str(w.message))
        t_idx = st.slider("Time index", 0, len(times) - 1, 0)
        fig = plot_stack_temperature(r_centres, z_centres, T[t_idx])
        st.pyplot(fig)


if __name__ == "__main__":
    main()
