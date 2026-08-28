"""Force-balance inversion: (gamma, chi, V, mu) -> (alpha, beta, T, quat_wb)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from f3p_attitude.aero import aero_force_body
from f3p_attitude.constants import (
    ALPHA_BETA_MAX,
    FSOLVE_XTOL,
    GRAVITY_WORLD,
    MASS,
    THRUST_MAX,
    THRUST_MIN,
)
from f3p_attitude.frames import mat_to_quat_xyzw, r_world_to_body
from f3p_attitude.kinematics import acceleration_world, velocity_world
from f3p_attitude.regime import MuMode, classify_mu_mode


def incidence_alpha(r_wb: np.ndarray, v_world: np.ndarray) -> float:
    """Geometric AoA: body-x vs velocity in the body x-z plane (rad, nose-up positive)."""
    v_body = r_wb @ v_world
    speed = np.linalg.norm(v_body)
    if speed < 1e-9:
        return 0.0
    return float(np.arctan2(v_body[2], v_body[0]))


@dataclass
class SolveSample:
    """One-time-step solution."""

    alpha: float  # geometric AoA (nose-up positive), for plots / logs
    alpha_rot: float  # Ry chain parameter in r_wind_to_body (warm-start x0[0])
    beta: float
    thrust: float
    quat_wb: np.ndarray  # [x, y, z, w], R_world_to_body
    mu_mode: MuMode
    r_wb: np.ndarray
    v_world: np.ndarray
    v_dot_world: np.ndarray
    residual: np.ndarray
    residual_norm: float
    success: bool
    message: str


@dataclass
class TrajectoryResult:
    t: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    thrust: np.ndarray
    quat_wb: np.ndarray  # (N, 4) xyzw
    mu_mode: np.ndarray  # object array of MuMode
    r_wb: np.ndarray
    residual_norm: np.ndarray
    success: np.ndarray


def force_residual_world(
    alpha: float,
    beta: float,
    thrust: float,
    gamma: float,
    chi: float,
    mu: float,
    v_world: np.ndarray,
    v_dot_world: np.ndarray,
) -> np.ndarray:
    """m * v_dot = F_aero_world + T * xb_world + m * g."""
    r_wb = r_world_to_body(gamma, chi, mu, alpha, beta)
    v_body = r_wb @ v_world
    f_body = aero_force_body(v_body, thrust)
    f_world = r_wb.T @ f_body
    xb_world = r_wb[0]
    f_total = f_world + thrust * xb_world
    return f_total + MASS * GRAVITY_WORLD - MASS * v_dot_world


def _alpha_beta_thrust_bounds() -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([-ALPHA_BETA_MAX, -ALPHA_BETA_MAX, THRUST_MIN])
    hi = np.array([ALPHA_BETA_MAX, ALPHA_BETA_MAX, THRUST_MAX])
    return lo, hi


def _static_hover_initial_guess(gamma: float) -> np.ndarray:
    """Nose-up guess when path gamma is near level (needs alpha_rot ~ -90 deg)."""
    if abs(gamma) < np.deg2rad(20.0):
        return np.array([-1.45, 0.0, MASS * 9.81 * 0.95])
    return np.array([0.05, 0.0, MASS * 9.81 * 0.85])


def static_hover_residual_world(
    alpha_rot: float,
    beta: float,
    thrust: float,
    gamma: float,
    chi: float,
    mu: float,
    v_dot_world: np.ndarray,
) -> np.ndarray:
    """
    V ~ 0: no relative wind, F_aero = 0. Balance T x_b + m g = m a (often a = 0).
    """
    r_wb = r_world_to_body(gamma, chi, mu, alpha_rot, beta)
    xb_world = r_wb[0]
    return thrust * xb_world + MASS * GRAVITY_WORLD - MASS * v_dot_world


def solve_static_hover(
    gamma: float,
    chi: float,
    mu: float,
    v_dot_world: np.ndarray | None = None,
    x0: np.ndarray | None = None,
) -> SolveSample:
    """True zero-speed hover: thrust along body x balances weight (no flat-plate loads)."""
    if v_dot_world is None:
        v_dot_world = np.zeros(3)
    v_world = np.zeros(3)

    def residual(x: np.ndarray) -> np.ndarray:
        return static_hover_residual_world(
            x[0], x[1], x[2], gamma, chi, mu, v_dot_world
        )

    x0_use = x0 if x0 is not None else _static_hover_initial_guess(gamma)
    bounds_lo, bounds_hi = _alpha_beta_thrust_bounds()

    sol = least_squares(
        residual,
        x0_use,
        bounds=(bounds_lo, bounds_hi),
        xtol=FSOLVE_XTOL,
        ftol=FSOLVE_XTOL,
    )
    alpha_rot, beta, thrust = sol.x
    r_wb = r_world_to_body(gamma, chi, mu, alpha_rot, beta)
    res = static_hover_residual_world(
        alpha_rot, beta, thrust, gamma, chi, mu, v_dot_world
    )
    xb = r_wb[0]
    body_pitch = float(np.arctan2(-xb[2], xb[0]))

    return SolveSample(
        alpha=body_pitch,
        alpha_rot=alpha_rot,
        beta=beta,
        thrust=thrust,
        quat_wb=mat_to_quat_xyzw(r_wb),
        mu_mode=MuMode.HOVER,
        r_wb=r_wb,
        v_world=v_world,
        v_dot_world=v_dot_world,
        residual=res,
        residual_norm=float(np.linalg.norm(res)),
        success=bool(sol.success),
        message=str(sol.message),
    )


def _initial_guess(mu_mode: MuMode, x0: np.ndarray | None) -> np.ndarray:
    if x0 is not None:
        return x0
    if mu_mode in (MuMode.VERTICAL, MuMode.HOVER):
        # Vertical / hover: nose roughly along velocity, moderate thrust
        return np.array([0.05, 0.0, MASS * 9.81 * 0.85])
    return np.array([0.12, 0.0, MASS * 9.81 * 0.85])


def solve_one(
    gamma: float,
    chi: float,
    speed: float,
    mu: float,
    gamma_dot: float = 0.0,
    chi_dot: float = 0.0,
    speed_dot: float = 0.0,
    x0: np.ndarray | None = None,
    mu_mode: MuMode | None = None,
) -> SolveSample:
    """
    Solve force balance for (alpha, beta, T) and build attitude as quaternion.

    mu is always roll about the instantaneous velocity vector (knife angle).
    Regime only selects the kinematic chart (cruise / vertical / hover).
    """
    if mu_mode is None:
        mu_mode = classify_mu_mode(gamma, speed)

    gamma_k, chi_k, speed_k = gamma, chi, speed
    v_world, _ = velocity_world(gamma_k, chi_k, speed_k)
    v_dot_world = acceleration_world(
        gamma_k, chi_k, speed_k, gamma_dot, chi_dot, speed_dot
    )

    x0_use = _initial_guess(mu_mode, x0)

    def residual(x: np.ndarray) -> np.ndarray:
        alpha, beta, thrust = x
        return force_residual_world(
            alpha,
            beta,
            thrust,
            gamma_k,
            chi_k,
            mu,
            v_world,
            v_dot_world,
        )

    bounds_lo, bounds_hi = _alpha_beta_thrust_bounds()

    sol = least_squares(
        residual,
        x0_use,
        bounds=(bounds_lo, bounds_hi),
        xtol=FSOLVE_XTOL,
        ftol=FSOLVE_XTOL,
    )

    alpha_rot, beta, thrust = sol.x
    r_wb = r_world_to_body(gamma_k, chi_k, mu, alpha_rot, beta)
    alpha = incidence_alpha(r_wb, v_world)
    quat_wb = mat_to_quat_xyzw(r_wb)
    res = force_residual_world(
        alpha_rot, beta, thrust, gamma_k, chi_k, mu, v_world, v_dot_world
    )

    return SolveSample(
        alpha=alpha,
        alpha_rot=alpha_rot,
        beta=beta,
        thrust=thrust,
        quat_wb=quat_wb,
        mu_mode=mu_mode,
        r_wb=r_wb,
        v_world=v_world,
        v_dot_world=v_dot_world,
        residual=res,
        residual_norm=float(np.linalg.norm(res)),
        success=bool(sol.success),
        message=str(sol.message),
    )


def solve_trajectory(
    t: np.ndarray,
    gamma: np.ndarray,
    chi: np.ndarray,
    speed: np.ndarray,
    mu: np.ndarray,
    gamma_dot: np.ndarray | None = None,
    chi_dot: np.ndarray | None = None,
    speed_dot: np.ndarray | None = None,
) -> TrajectoryResult:
    n = len(t)
    if gamma_dot is None:
        gamma_dot = np.gradient(gamma, t)
    if chi_dot is None:
        chi_dot = np.gradient(chi, t)
    if speed_dot is None:
        speed_dot = np.gradient(speed, t)

    alpha = np.zeros(n)
    beta = np.zeros(n)
    thrust = np.zeros(n)
    quat_wb = np.zeros((n, 4))
    mu_mode = np.empty(n, dtype=object)
    r_wb = np.zeros((n, 3, 3))
    residual_norm = np.zeros(n)
    success = np.zeros(n, dtype=bool)

    x0 = None
    for i in range(n):
        sample = solve_one(
            gamma[i],
            chi[i],
            speed[i],
            mu[i],
            gamma_dot[i],
            chi_dot[i],
            speed_dot[i],
            x0=x0,
        )
        alpha[i] = sample.alpha
        beta[i] = sample.beta
        thrust[i] = sample.thrust
        quat_wb[i] = sample.quat_wb
        mu_mode[i] = sample.mu_mode
        r_wb[i] = sample.r_wb
        residual_norm[i] = sample.residual_norm
        success[i] = sample.success
        x0 = np.array([sample.alpha_rot, sample.beta, sample.thrust])

    return TrajectoryResult(
        t=t,
        alpha=alpha,
        beta=beta,
        thrust=thrust,
        quat_wb=quat_wb,
        mu_mode=mu_mode,
        r_wb=r_wb,
        residual_norm=residual_norm,
        success=success,
    )
