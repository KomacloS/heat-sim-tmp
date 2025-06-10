"""Beam-shape transient demo for Milestone 3."""

from __future__ import annotations

import streamlit as st
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import EngFormatter

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
    dt_ms = st.number_input("Time step (ms)", value=0.1, format="%.6f")
    max_iter = st.number_input(
        "Max iterations per step", value=1000, min_value=1, step=1
    )
    allow_unstable = st.checkbox("Ignore stability limit")

    if dt_ms < 0.01:
        st.warning("Time step is very small; simulation may be slow and not optimal.")

    t_max = t_max_ms / 1000.0
    dt = dt_ms / 1000.0

    if "m3_results" not in st.session_state:
        st.session_state["m3_results"] = None

    run = st.button("Run")
    if run:
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

        try:
            times, T = solve_transient(
                r_centres,
                dr,
                0.0,
                k,
                rho_cp,
                t_max,
                dt,
                src,
                max_steps=int(max_iter),
                allow_unstable=allow_unstable,
                progress_cb=cb,
            )
        except ValueError as exc:
            progress.empty()
            status.error(str(exc))
            return
        progress.empty()
        status.success(f"Completed in {time.perf_counter() - start:.1f}s")
        st.session_state["m3_results"] = (r_centres, times, T, beam_type)

    if st.session_state["m3_results"] is not None:
        r_centres, times, T, beam_type = st.session_state["m3_results"]

        r_centres_mm = r_centres * 1000.0
        time_ms = st.slider(
            "Time (ms)",
            min_value=0.0,
            max_value=float(times[-1] * 1000),
            value=0.0,
            step=dt_ms,
            format="%.3f",
        )
        t_idx = min(int(round(time_ms / dt_ms)), len(times) - 1)

        fig, ax = plt.subplots()
        ax.plot(r_centres_mm, T[t_idx, :])
        ax.set_xlabel("Radius (mm)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"Beam: {beam_type}, t = {times[t_idx]*1000:.1f} ms")
        ax.xaxis.set_major_formatter(EngFormatter(unit="mm"))
        ax.yaxis.set_major_formatter(EngFormatter(unit="°C"))
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        tt, rr = np.meshgrid(times * 1000.0, r_centres_mm)
        pcm = ax2.pcolormesh(tt, rr, T.T, shading="auto")
        cbar = fig2.colorbar(pcm, ax=ax2, label="Temperature (°C)")
        cbar.formatter = EngFormatter(unit="°C")
        cbar.update_ticks()
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Radius (mm)")
        ax2.xaxis.set_major_formatter(EngFormatter(unit="ms"))
        ax2.yaxis.set_major_formatter(EngFormatter(unit="mm"))
        ax2.set_title("Temperature vs. time")
        st.pyplot(fig2)

        theta = np.linspace(0.0, 2 * np.pi, 200)
        th, rr2 = np.meshgrid(theta, r_centres_mm)
        temp_ring = np.tile(T[t_idx, :], (len(theta), 1))
        fig3 = plt.figure(figsize=(4, 4))
        ax3 = fig3.add_subplot(111, projection="polar")
        pcm2 = ax3.pcolormesh(th, rr2, temp_ring.T, shading="auto")
        cbar2 = fig3.colorbar(pcm2, ax=ax3, label="Temperature (°C)")
        cbar2.formatter = EngFormatter(unit="°C")
        cbar2.update_ticks()
        ax3.set_title("Radial temperature")
        ax3.set_yticklabels([])
        st.pyplot(fig3)


if __name__ == "__main__":
    main()
