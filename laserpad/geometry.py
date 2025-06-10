"""Pad geometry utilities."""

from __future__ import annotations

import math
from typing import Dict, cast, List, Tuple
import json
from pathlib import Path
import yaml  # type: ignore
from numpy.typing import NDArray

import numpy as np


def get_annular_pad_properties(
    r_inner_mm: float,
    r_outer_mm: float,
    thickness_mm: float = 0.035,
    density: float = 8960.0,
    cp: float = 385.0,
) -> dict[str, float]:
    """Return area, volume, mass and heat capacity for an annular pad."""

    r_in = r_inner_mm * 1e-3
    r_out = r_outer_mm * 1e-3
    t = thickness_mm * 1e-3

    area = math.pi * (r_out**2 - r_in**2)
    volume = area * t
    mass = density * volume
    heat_capacity = mass * cp

    return {
        "area_m2": area,
        "volume_m3": volume,
        "mass_kg": mass,
        "heat_capacity_J_per_K": heat_capacity,
    }


def get_pad_properties(
    diameter_mm: float,
    thickness_mm: float = 0.035,
    density: float = 8960.0,
    cp: float = 385.0,
) -> Dict[str, float]:
    """Return properties for a solid circular pad.

    This function is retained for backward compatibility and simply calls
    :func:`get_annular_pad_properties` with ``r_inner_mm`` set to 0.
    """

    r_out = diameter_mm / 2.0
    return get_annular_pad_properties(
        0.0,
        r_out,
        thickness_mm,
        density,
        cp,
    )


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


def load_materials(path: str = "materials.yaml") -> Dict[str, Dict[str, float]]:
    """Return material properties dictionary from a YAML file."""
    data = yaml.safe_load(Path(path).read_text())
    return cast(Dict[str, Dict[str, float]], data)


def build_stack_mesh(
    r_inner: float,
    r_outer: float,
    n_r: int,
    pad_th: float,
    sub_th: float,
    n_z: int,
) -> tuple[NDArray[np.float_], float, NDArray[np.float_], float, NDArray[np.str_]]:
    """Return 2-D r-z mesh centres and material index grid."""

    dr = (r_outer - r_inner) / n_r
    dz = (pad_th + sub_th) / n_z

    r_centres = r_inner + (np.arange(n_r) + 0.5) * dr
    z_centres = (np.arange(n_z) + 0.5) * dz

    mat_idx = np.full((n_z, n_r), "fr4", dtype=object)
    pad_cells = z_centres < pad_th
    mat_idx[pad_cells, :] = "copper"

    return r_centres, dr, z_centres, dz, mat_idx


def load_traces(path: str | bytes | object) -> List[Tuple[float, float]]:
    """Load a JSON file describing radial copper trace angles."""
    if hasattr(path, "read"):
        text = path.read()
        if isinstance(text, bytes):
            text = text.decode()
    else:
        text = Path(str(path)).read_text()
    data = json.loads(text)
    return [(d["start_angle"], d["end_angle"]) for d in data]


def build_stack_mesh_with_traces(
    r_inner: float,
    r_outer: float,
    n_r: int,
    pad_th: float,
    sub_th: float,
    n_z: int,
    trace_defs: List[Tuple[float, float]],
    n_theta: int = 360,
) -> tuple[
    NDArray[np.float_],
    float,
    NDArray[np.float_],
    float,
    NDArray[np.str_],
    NDArray[np.bool_],
]:
    """Return mesh plus boolean trace mask for each angular cell."""

    r_centres, dr, z_centres, dz, mat_idx = build_stack_mesh(
        r_inner, r_outer, n_r, pad_th, sub_th, n_z
    )

    theta_centres = np.linspace(0.0, 360.0, n_theta, endpoint=False)
    trace_mask = np.zeros((n_theta, n_r), dtype=bool)

    for k, theta in enumerate(theta_centres):
        for start, end in trace_defs:
            if start <= theta < end or (
                end < start and (theta >= start or theta < end)
            ):
                trace_mask[k, :] = True
                break

    return r_centres, dr, z_centres, dz, mat_idx, trace_mask
