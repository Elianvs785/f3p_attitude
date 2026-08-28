"""Flat-plate aerodynamics on physical_model.md geometry."""

import numpy as np

from f3p_attitude.constants import PROP_DISK_AREA, RHO
from f3p_attitude.surfaces import Surface, f3p_surfaces


def rotate_normal(n_surface: np.ndarray, delta: float, hinge_axis: np.ndarray) -> np.ndarray:
    h = hinge_axis / np.linalg.norm(hinge_axis)
    c, s = np.cos(delta), np.sin(delta)
    return (
        n_surface * c
        + np.cross(h, n_surface) * s
        + h * np.dot(h, n_surface) * (1.0 - c)
    )


def flat_plate_force(
    v_local: np.ndarray,
    area: float,
    n_surface_0: np.ndarray,
    delta: float = 0.0,
    hinge_axis: np.ndarray | None = None,
) -> np.ndarray:
    v_norm = np.linalg.norm(v_local)
    if v_norm < 1e-6:
        return np.zeros(3)

    if hinge_axis is None:
        n_eff = n_surface_0
    else:
        n_eff = rotate_normal(n_surface_0, delta, hinge_axis)

    e_v = v_local / v_norm
    # sin > 0 when flow meets the plate from the normal side (n = body -z, upright AoA)
    sin_alpha = np.clip(np.dot(e_v, n_eff), -1.0, 1.0)
    q_dyn = 0.5 * RHO * v_norm**2 * area
    c_n = 2.0 * sin_alpha
    return -c_n * q_dyn * n_eff


def induced_velocity(thrust: float, v_axial: float) -> float:
    """Actuator-disk induced speed (non-negative)."""
    if thrust <= 0.0:
        return 0.0
    a = 2.0 * RHO * PROP_DISK_AREA
    b = 2.0 * RHO * PROP_DISK_AREA * max(v_axial, 0.0)
    c = -thrust
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return 0.0
    return max((-b + np.sqrt(disc)) / (2.0 * a), 0.0)


def local_velocity_body(
    v_body: np.ndarray,
    omega_body: np.ndarray,
    r_surface: np.ndarray,
    in_propwash: bool,
    thrust: float,
) -> np.ndarray:
    v_local = v_body + np.cross(omega_body, r_surface)
    if in_propwash:
        v_i = induced_velocity(thrust, v_body[0])
        v_local = v_local.copy()
        v_local[0] += 2.0 * v_i
    return v_local


def parasitic_drag_body(
    v_body: np.ndarray, s_ref: float, cd_axial: float
) -> np.ndarray:
    """Small drag along local velocity (flat plate has no chordwise force)."""
    speed = np.linalg.norm(v_body)
    if speed < 1e-6:
        return np.zeros(3)
    q = 0.5 * RHO * speed**2
    ehat = v_body / speed
    return -cd_axial * q * s_ref * ehat


def aero_force_body(
    v_body: np.ndarray,
    thrust: float,
    omega_body: np.ndarray | None = None,
    surfaces: list[Surface] | None = None,
    cd_axial: float = 0.12,
) -> np.ndarray:
    if omega_body is None:
        omega_body = np.zeros(3)
    if surfaces is None:
        surfaces = f3p_surfaces()

    f_total = np.zeros(3)
    s_ref = 0.0
    for surf in surfaces:
        s_ref += surf.area
        v_loc = local_velocity_body(
            v_body, omega_body, surf.r_body, surf.in_propwash, thrust
        )
        f_total += flat_plate_force(
            v_loc,
            surf.area,
            surf.normal,
            delta=0.0,
            hinge_axis=surf.hinge_axis,
        )
    f_total += parasitic_drag_body(v_body, s_ref, cd_axial)
    return f_total
