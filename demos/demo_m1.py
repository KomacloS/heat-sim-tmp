"""Streamlit demo for Milestone 1."""

from __future__ import annotations

import streamlit as st

from laserpad.geometry import get_pad_properties
from laserpad.solver import solve_heatup
from laserpad.plot import plot_heatup


def main() -> None:
    st.title("M1: Lumped-Pad Heatup Demo")

    d_mm = st.slider(
        "Pad diameter (mm)", min_value=0.5, max_value=5.0, value=1.0, step=0.1
    )
    th_mm = st.number_input("Pad thickness (mm)", value=0.035, step=0.005)
    power_mW = st.slider("Laser power (W)", 10.0, 10000.0, 1000.0, step=10.0)
    t_max = st.slider("Total time (s)", 0.1, 5.0, 1.0, step=0.1)
    dt = st.slider("Time step (s)", 0.001, 0.1, 0.02, step=0.001)
    T0 = st.number_input("Initial temperature (°C)", value=25.0, step=1.0)

    props = get_pad_properties(d_mm, th_mm)
    times, temps = solve_heatup(
        power_mW, props["mass_kg"], props["heat_capacity_J_per_K"], t_max, dt, T0
    )

    fig = plot_heatup(times, temps)
    st.pyplot(fig)

    st.markdown(
        f"""
**Pad mass:** {props['mass_kg']:.4f} kg  
**Heat capacity:** {props['heat_capacity_J_per_K']:.1f} J/K  
**Peak ΔT:** {(temps[-1]-T0):.1f} °C
"""
    )


if __name__ == "__main__":
    main()
