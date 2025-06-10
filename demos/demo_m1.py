"""Streamlit demo for Milestone 1."""

from __future__ import annotations

import streamlit as st
import time
from matplotlib.ticker import EngFormatter

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
    dt = st.slider("Time step (s)", 0.001, 0.1, 0.02, step=0.001, format="%.6f")
    max_iter = st.number_input(
        "Max iterations per step", min_value=1000, value=10000
    )
    T0 = st.number_input("Initial temperature (°C)", value=25.0, step=1.0)

    props = get_pad_properties(d_mm, th_mm)
    progress = st.progress(0)
    status = st.empty()
    start = time.perf_counter()

    def cb(i: int, total: int) -> None:
        frac = i / total
        progress.progress(int(frac * 100))
        elapsed = time.perf_counter() - start
        est_total = elapsed / frac if frac else 0.0
        remaining = est_total - elapsed
        status.text(
            f"Iteration {i}/{total} — elapsed {elapsed:.1f}s, ETA {remaining:.1f}s"
        )

    times, temps = solve_heatup(
        power_mW,
        props["mass_kg"],
        props["heat_capacity_J_per_K"],
        t_max,
        dt,
        T0,
        max_steps=int(max_iter),
        progress_cb=cb,
    )
    progress.empty()
    status.success(f"Completed in {time.perf_counter() - start:.1f}s")

    fig = plot_heatup(times, temps)
    st.pyplot(fig)

    eng_temp = EngFormatter(unit="°C", places=2)
    eng_mass = EngFormatter(unit="kg", places=2)
    eng_cp = EngFormatter(unit="J/K", places=2)
    st.markdown(
        f"""
**Pad mass:** {eng_mass.format_data(props['mass_kg'])}
**Heat capacity:** {eng_cp.format_data(props['heat_capacity_J_per_K'])}
**Peak ΔT:** {eng_temp.format_data(temps[-1] - T0)}
"""
    )


if __name__ == "__main__":
    main()
