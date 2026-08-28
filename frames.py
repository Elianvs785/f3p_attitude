"""Rotation matrices (passive): v_B = R_A_to_B @ v_A; quaternion helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def velocity_unit_ned(gamma: float, chi: float) -> np.ndarray:
    """Unit velocity in NED; gamma>0 climb, chi from north (x) CCW."""
    cg, sg = np.cos(gamma), np.sin(gamma)
    cc, sc = np.cos(chi), np.sin(chi)
    return np.array([cg * cc, cg * sc, -sg])


def dv_hat_dgamma(gamma: float, chi: float) -> np.ndarray:
    cg, sg = np.cos(gamma), np.sin(gamma)
    cc, sc = np.cos(chi), np.sin(chi)
    return np.array([-sg * cc, -sg * sc, -cg])


def dv_hat_dchi(gamma: float, chi: float) -> np.ndarray:
    cg = np.cos(gamma)
    cc, sc = np.cos(chi), np.sin(chi)
    return np.array([-cg * sc, cg * cc, 0.0])


def r_world_to_wind(gamma: float, chi: float) -> np.ndarray:
    """Passive R: world -> wind (velocity-aligned frame)."""
    return rot_y(-gamma) @ rot_z(-chi)


def r_wind_to_body(mu: float, alpha: float, beta: float) -> np.ndarray:
    """Intrinsic ZYX in wind frame: beta, alpha, mu (see attitude_trajectory_generator_tutorial)."""
    return rot_x(mu) @ rot_y(alpha) @ rot_z(beta)


def r_world_to_body(
    gamma: float, chi: float, mu: float, alpha: float, beta: float
) -> np.ndarray:
    r_w_wind = r_world_to_wind(gamma, chi)
    r_w_body = r_wind_to_body(mu, alpha, beta)
    return r_w_body @ r_w_wind


def body_axes_world(r_wb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows of R_wb are body basis vectors in world frame."""
    return r_wb[0], r_wb[1], r_wb[2]


def mat_to_quat_xyzw(r_wb: np.ndarray) -> np.ndarray:
    """Unit quaternion [x, y, z, w] from R_world_to_body (scipy convention)."""
    return Rotation.from_matrix(r_wb).as_quat()


def quat_xyzw_to_mat(quat: np.ndarray) -> np.ndarray:
    """R_world_to_body from [x, y, z, w]."""
    return Rotation.from_quat(quat).as_matrix()
