"""Trace-aware multilayer demo for Milestone 5."""

from __future__ import annotations

import streamlit as st

from laserpad.geometry import load_traces, build_stack_mesh_with_traces
from laserpad.solver import solve_transient_2d
from laserpad.plot import plot_stack_temperature


def main() -> None:
    st.title("M5: Multilayer + Traces")

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

    trace_file = st.file_uploader("Trace JSON config", type="json")
    h_trace = st.number_input("Trace h (W/m²·K)", value=1e3)
    T_inf = st.number_input("Ambient T (°C)", value=25.0)

    if st.button("Run") and trace_file is not None:
        traces = load_traces(trace_file)
        r, dr, z, dz, mat_idx, mask = build_stack_mesh_with_traces(
            r_in, r_out, n_r, pad_th, sub_th, n_z, traces
        )
        height = pad_th + sub_th
        q_flux = power_W / (2.0 * np.pi * r_in * height)
        times, T = solve_transient_2d(
            r,
            dr,
            z,
            dz,
            mat_idx,
            q_flux,
            n_t,
            dt,
            trace_mask=mask,
            h_trace=h_trace,
            T_inf=T_inf,
        )
        t_idx = st.slider("Time index", 0, len(times) - 1, 0)
        fig = plot_stack_temperature(r, z, T[t_idx])
        st.pyplot(fig)


if __name__ == "__main__":
    main()
