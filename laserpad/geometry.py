"""Pad geometry utilities."""

from __future__ import annotations

import math
from typing import Dict
from numpy.typing import NDArray

import numpy as np


def get_pad_properties(
    diameter_mm: float,
    thickness_mm: float = 0.035,
    density: float = 8960.0,
    cp: float = 385.0,
) -> Dict[str, float]:
    """Return area, volume, mass and heat capacity for a circular pad."""
    diameter_m = diameter_mm / 1000.0
    thickness_m = thickness_mm / 1000.0

    area_m2 = math.pi * (diameter_m / 2.0) ** 2
    volume_m3 = area_m2 * thickness_m
    mass_kg = density * volume_m3
    heat_capacity_J_per_K = mass_kg * cp

    return {
        "area_m2": area_m2,
        "volume_m3": volume_m3,
        "mass_kg": mass_kg,
        "heat_capacity_J_per_K": heat_capacity_J_per_K,
    }


def build_radial_mesh(
    r_inner_m: float, r_outer_m: float, n_r: int
) -> tuple[NDArray[np.float_], float]:
    """Returns (r_centres, dr) for a uniform 1-D cylindrical mesh."""
    if n_r <= 0:
        raise ValueError("n_r must be positive")
    if r_outer_m <= r_inner_m:
        raise ValueError("r_outer_m must be larger than r_inner_m")

    dr = (r_outer_m - r_inner_m) / n_r
    r_centres = r_inner_m + (np.arange(n_r) + 0.5) * dr
    return r_centres, dr
