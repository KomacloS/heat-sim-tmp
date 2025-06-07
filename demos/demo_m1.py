# demos/demo_m1.py

import argparse

import matplotlib.pyplot as plt

from laserpad.geometry import build_mesh
from laserpad.solver import solve_steady
from laserpad.plot import plot_ring


def main():
    parser = argparse.ArgumentParser(
        description="M1 Demo: Steady-state temperature on a copper pad ring."
    )
    parser.add_argument(
        "--r-inner",
        type=float,
        default=0.50,
        help="Inner pad radius (default: 0.50)",
    )
    parser.add_argument(
        "--r-outer",
        type=float,
        default=1.50,
        help="Outer pad radius (default: 1.50)",
    )
    parser.add_argument(
        "--q-flux",
        type=float,
        default=1.0e6,
        help="Heat flux at inner rim (default: 1e6)",
    )
    parser.add_argument(
        "--n-r",
        type=int,
        default=200,
        help="Number of radial cells in mesh (default: 200)",
    )

    args = parser.parse_args()
    r_inner = args.r_inner
    r_outer = args.r_outer
    q_flux = args.q_flux
    n_r = args.n_r

    mesh = build_mesh(r_inner=r_inner, r_outer=r_outer, n_r=n_r)
    temperature = solve_steady(mesh=mesh, q_inner=q_flux, k=400.0, r_outer=r_outer)
    fig = plot_ring(mesh=mesh, temperature=temperature, r_inner=r_inner, r_outer=r_outer)

    plt.show()


if __name__ == "__main__":
    main()
