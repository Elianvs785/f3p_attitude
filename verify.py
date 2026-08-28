"""Checks that solver outputs satisfy force balance and physical bounds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from f3p_attitude.constants import (
    MASS,
    RESIDUAL_TOL,
    THRUST_MAX,
    THRUST_MIN,
)
from f3p_attitude.frames import velocity_unit_ned
from f3p_attitude.regime import MuMode
from f3p_attitude.solver import SolveSample, solve_one, solve_static_hover, solve_trajectory
from f3p_attitude.scenarios import level_to_vertical_profile
from scipy.spatial.transform import Rotation


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_residual(sample: SolveSample, tol: float = RESIDUAL_TOL) -> CheckResult:
    ok = sample.residual_norm < tol and sample.success
    return CheckResult(
        "residual",
        ok,
        f"||r||={sample.residual_norm:.2e}, success={sample.success}",
    )


def check_thrust(sample: SolveSample) -> CheckResult:
    ok = THRUST_MIN <= sample.thrust <= THRUST_MAX
    return CheckResult(
        "thrust_bounds",
        ok,
        f"T={sample.thrust:.4f} N in [{THRUST_MIN}, {THRUST_MAX}]",
    )


def check_quaternion(sample: SolveSample, tol: float = 1e-8) -> CheckResult:
    q = sample.quat_wb
    norm_err = abs(np.linalg.norm(q) - 1.0)
    ok = norm_err < tol
    return CheckResult(
        "quaternion_unit",
        ok,
        f"|q|={np.linalg.norm(q):.6f}, err={norm_err:.2e}",
    )


def check_rotation(sample: SolveSample, orth_tol: float = 1e-6) -> CheckResult:
    r = sample.r_wb
    orth_err = np.linalg.norm(r @ r.T - np.eye(3))
    det_err = abs(np.linalg.det(r) - 1.0)
    ok = orth_err < orth_tol and det_err < orth_tol
    return CheckResult(
        "rotation_matrix",
        ok,
        f"orth_err={orth_err:.2e}, det_err={det_err:.2e}",
    )


def check_nose_along_velocity(sample: SolveSample, tol: float = 0.95) -> CheckResult:
    v_hat = sample.v_world / max(np.linalg.norm(sample.v_world), 1e-9)
    align = float(np.dot(sample.r_wb[0], v_hat))
    ok = align > tol
    return CheckResult(
        "nose_aligns_velocity",
        ok,
        f"xb dot v_hat = {align:.4f}",
    )


def check_sample(sample: SolveSample) -> list[CheckResult]:
    return [
        check_residual(sample),
        check_thrust(sample),
        check_quaternion(sample),
        check_rotation(sample),
        check_nose_along_velocity(sample),
    ]


def check_steady_level_coordinated(
    speed: float = 5.0, mu: float = 0.0
) -> list[CheckResult]:
    sample = solve_one(
        gamma=0.0,
        chi=0.0,
        speed=speed,
        mu=mu,
        gamma_dot=0.0,
        chi_dot=0.0,
        speed_dot=0.0,
    )
    results = check_sample(sample)
    results.append(
        CheckResult(
            "mu_mode_cruise",
            sample.mu_mode == MuMode.CRUISE,
            f"mu_mode={sample.mu_mode.value}",
        )
    )
    results.append(
        CheckResult(
            "beta_near_zero",
            abs(sample.beta) < np.deg2rad(5.0),
            f"beta={np.rad2deg(sample.beta):.2f} deg",
        )
    )
    weight = MASS * 9.81
    results.append(
        CheckResult(
            "thrust_near_weight",
            0.3 * weight < sample.thrust < 2.5 * weight,
            f"T={sample.thrust:.3f} N, mg={weight:.3f} N",
        )
    )
    return results


def check_steady_knife(speed: float = 5.0, mu_deg: float = 90.0) -> list[CheckResult]:
    mu = np.deg2rad(mu_deg)
    level = solve_one(0.0, 0.0, speed, 0.0)
    knife = solve_one(0.0, 0.0, speed, mu)
    results = check_sample(knife)
    results.append(
        CheckResult(
            "knife_differs_from_level",
            np.linalg.norm(knife.r_wb - level.r_wb) > 0.1,
            f"||R_knife - R_level||={np.linalg.norm(knife.r_wb - level.r_wb):.3f}",
        )
    )
    return results


def check_vertical_constant_mu(
    speed: float = 5.0, mu_deg: float = 30.0
) -> list[CheckResult]:
    """Vertical climb, mu fixed: attitude constant, nose up along velocity."""
    gamma = np.deg2rad(90.0)
    mu = np.deg2rad(mu_deg)
    s0 = solve_one(gamma, 0.0, speed, mu)
    s1 = solve_one(
        gamma, 0.0, speed, mu, x0=np.array([s0.alpha_rot, s0.beta, s0.thrust])
    )

    results = check_sample(s0)
    results.append(
        CheckResult(
            "mu_mode_vertical",
            s0.mu_mode == MuMode.VERTICAL,
            f"mu_mode={s0.mu_mode.value}",
        )
    )
    results.append(
        CheckResult(
            "repeatable_solution",
            abs(s1.alpha - s0.alpha) < 1e-4
            and abs(s1.beta - s0.beta) < 1e-4
            and abs(s1.thrust - s0.thrust) < 1e-3,
            f"alpha,beta,T stable on re-solve",
        )
    )
    v_hat = velocity_unit_ned(gamma, 0.0)
    climb_up = v_hat[2] < 0
    results.append(
        CheckResult(
            "velocity_points_up_ned",
            climb_up,
            f"v_hat NED z={v_hat[2]:.3f} (negative z = up)",
        )
    )
    quat_spread = np.linalg.norm(s1.quat_wb - s0.quat_wb)
    results.append(
        CheckResult(
            "quat_stable_constant_mu",
            quat_spread < 1e-6,
            f"|dq|={quat_spread:.2e}",
        )
    )
    return results


def check_vertical_mu_ramp(
    speed: float = 5.0,
    duration: float = 2.0,
    mu_rate_deg_s: float = 25.0,
) -> list[CheckResult]:
    """Vertical climb, mu increases linearly: quat must change, force balance held."""
    t = np.linspace(0.0, duration, 21)
    gamma = np.full_like(t, np.deg2rad(90.0))
    mu_rate = np.deg2rad(mu_rate_deg_s)
    mu = mu_rate * t

    traj = solve_trajectory(t, gamma, np.zeros_like(t), np.full_like(t, speed), mu)
    results = check_trajectory(traj)
    results.append(
        CheckResult(
            "all_vertical_mode",
            all(m == MuMode.VERTICAL for m in traj.mu_mode),
            "mu_mode vertical at all samples",
        )
    )

    dq = np.linalg.norm(np.diff(traj.quat_wb, axis=0), axis=1)
    results.append(
        CheckResult(
            "quat_changes_with_mu_ramp",
            bool(np.all(dq > 1e-4)),
            f"mean |dq|={float(np.mean(dq)):.4f}, min={float(np.min(dq)):.4f}",
        )
    )
    mu_fit = np.polyfit(t, mu, 1)
    results.append(
        CheckResult(
            "mu_linear_in_time",
            abs(mu_fit[0] - mu_rate) < np.deg2rad(0.5),
            f"fitted mu_dot={np.rad2deg(mu_fit[0]):.2f} deg/s, cmd={mu_rate_deg_s}",
        )
    )
    return results


def check_chi_invariance(speed: float = 5.0, delta_chi: float = 0.5) -> list[CheckResult]:
    mu = np.deg2rad(30.0)
    s0 = solve_one(0.0, 0.0, speed, mu)
    s1 = solve_one(
        0.0, delta_chi, speed, mu, x0=np.array([s0.alpha_rot, s0.beta, s0.thrust])
    )

    t_ok = abs(s1.thrust - s0.thrust) < 1e-3
    ab_ok = (
        abs(s1.alpha - s0.alpha) < 1e-4 and abs(s1.beta - s0.beta) < 1e-4
    )

    return [
        CheckResult("chi_invariance_T", t_ok, f"T0={s0.thrust:.4f}, T1={s1.thrust:.4f}"),
        CheckResult(
            "chi_invariance_alpha_beta",
            ab_ok,
            f"alpha0={s0.alpha:.4f}, alpha1={s1.alpha:.4f}, "
            f"beta0={np.rad2deg(s0.beta):.2f} deg",
        ),
    ]


def check_level_to_vertical_maneuver(
    dt: float = 0.1,
    level_duration: float = 4.0,
    blend_duration: float = 4.0,
    vertical_hold: float = 4.0,
    speed: float = 5.0,
) -> list[CheckResult]:
    """
    End-to-end: level cruise -> smooth pitch to vertical -> vertical climb.
    Validates cruise, transition, and vertical segments with quaternion output.
    """
    t_end = level_duration + blend_duration + vertical_hold
    t = np.arange(0.0, t_end + dt * 0.5, dt)
    profile = level_to_vertical_profile(
        t,
        level_duration=level_duration,
        blend_duration=blend_duration,
        vertical_hold=vertical_hold,
        speed=speed,
        mu=0.0,
    )

    traj = solve_trajectory(
        t,
        profile["gamma"],
        profile["chi"],
        profile["speed"],
        profile["mu"],
        gamma_dot=profile["gamma_dot"],
        chi_dot=profile["chi_dot"],
        speed_dot=profile["speed_dot"],
    )
    results = check_trajectory(traj)

    t_vert = float(profile["vertical_start"][0])
    mask_level = t <= level_duration * 0.95
    mask_vert = t >= t_vert + 0.05 * vertical_hold
    mask_blend = (~mask_level) & (~mask_vert)

    modes_level = [traj.mu_mode[i] for i in np.where(mask_level)[0]]
    modes_vert = [traj.mu_mode[i] for i in np.where(mask_vert)[0]]

    results.append(
        CheckResult(
            "phase_level_is_cruise",
            len(modes_level) > 0 and all(m == MuMode.CRUISE for m in modes_level),
            f"cruise samples={len(modes_level)}",
        )
    )
    results.append(
        CheckResult(
            "phase_vertical_is_vertical_mode",
            len(modes_vert) > 0 and all(m == MuMode.VERTICAL for m in modes_vert),
            f"vertical samples={len(modes_vert)}",
        )
    )
    results.append(
        CheckResult(
            "phase_blend_nonempty",
            int(np.sum(mask_blend)) >= 5,
            f"blend samples={int(np.sum(mask_blend))}",
        )
    )

    # Quaternion continuity (no gimbal-lock jumps)
    rots = Rotation.from_quat(traj.quat_wb)
    step_angles = []
    for i in range(len(t) - 1):
        rel = rots[i].inv() * rots[i + 1]
        step_angles.append(rel.magnitude())
    max_step = float(np.max(step_angles))
    results.append(
        CheckResult(
            "quat_continuous",
            max_step < np.deg2rad(25.0),
            f"max inter-step angle={np.rad2deg(max_step):.2f} deg",
        )
    )

    # Integrated path shape (NED): north then up
    v_hat = np.array(
        [
            velocity_unit_ned(profile["gamma"][i], profile["chi"][i])
            for i in range(len(t))
        ]
    )
    p = np.zeros((len(t), 3))
    for i in range(len(t) - 1):
        dt_i = t[i + 1] - t[i]
        v_i = profile["speed"][i] * v_hat[i]
        v_ip1 = profile["speed"][i + 1] * v_hat[i + 1]
        p[i + 1] = p[i] + 0.5 * (v_i + v_ip1) * dt_i

    north = p[-1, 0] - p[0, 0]
    up_ned = -(p[-1, 2] - p[0, 2])
    results.append(
        CheckResult(
            "path_north_then_up",
            north > 15.0 and up_ned > 15.0,
            f"delta_north={north:.1f} m, delta_up={up_ned:.1f} m",
        )
    )

    # Nose along velocity in each phase (spot check mid indices)
    def _nose_ok(indices: np.ndarray) -> float:
        aligns = []
        for i in indices:
            v = profile["speed"][i] * v_hat[i]
            aligns.append(float(np.dot(traj.r_wb[i, 0], v / np.linalg.norm(v))))
        return float(np.min(aligns)) if aligns else 0.0

    i_mid_level = np.where(mask_level)[0][len(np.where(mask_level)[0]) // 2]
    i_mid_vert = np.where(mask_vert)[0][len(np.where(mask_vert)[0]) // 2]
    i_mid_blend = np.where(mask_blend)[0][len(np.where(mask_blend)[0]) // 2]

    for name, idx in [
        ("level", np.array([i_mid_level])),
        ("blend", np.array([i_mid_blend])),
        ("vertical", np.array([i_mid_vert])),
    ]:
        a = _nose_ok(idx)
        results.append(
            CheckResult(
                f"nose_align_{name}",
                a > 0.9,
                f"min xb dot v_hat = {a:.4f}",
            )
        )

    results.append(
        CheckResult(
            "gamma_reaches_vertical",
            float(np.max(np.rad2deg(profile["gamma"][mask_vert]))) > 89.0,
            f"gamma_vert max={np.rad2deg(profile['gamma'][mask_vert].max()):.1f} deg",
        )
    )

    return results


def check_static_hover_vertical() -> list[CheckResult]:
    """V=0, gamma=90: thrust balances weight, nose up."""
    s = solve_static_hover(np.deg2rad(90.0), 0.0, 0.0)
    xb = s.r_wb[0]
    mg = MASS * 9.81
    return [
        CheckResult(
            "hover_residual",
            s.residual_norm < RESIDUAL_TOL,
            f"||r||={s.residual_norm:.2e}",
        ),
        CheckResult(
            "hover_thrust_mg",
            abs(s.thrust - mg) < 0.02,
            f"T={s.thrust:.3f} N, mg={mg:.3f} N",
        ),
        CheckResult(
            "hover_nose_up",
            xb[2] < -0.99,
            f"xb NED z={xb[2]:.3f} (negative = up)",
        ),
    ]


def check_trajectory(traj) -> list[CheckResult]:
    max_res = float(np.max(traj.residual_norm))
    all_ok = bool(np.all(traj.success))
    thrust_ok = bool(
        np.all((traj.thrust >= THRUST_MIN) & (traj.thrust <= THRUST_MAX))
    )
    return [
        CheckResult(
            "trajectory_residual",
            max_res < RESIDUAL_TOL and all_ok,
            f"max ||r||={max_res:.2e}, all success={all_ok}",
        ),
        CheckResult("trajectory_thrust", thrust_ok, "thrust in bounds at all t"),
    ]


def run_all() -> bool:
    suites = [
        ("steady_level_mu0", check_steady_level_coordinated()),
        ("steady_knife_mu90", check_steady_knife(mu_deg=90.0)),
        ("vertical_constant_mu", check_vertical_constant_mu()),
        ("vertical_mu_ramp", check_vertical_mu_ramp()),
        ("level_to_vertical_maneuver", check_level_to_vertical_maneuver()),
        ("chi_invariance", check_chi_invariance()),
        ("static_hover_vertical", check_static_hover_vertical()),
    ]

    t = np.linspace(0.0, 2.0, 41)
    traj = solve_trajectory(
        t,
        np.zeros_like(t),
        np.zeros_like(t),
        np.full_like(t, 5.0),
        np.zeros_like(t),
    )
    suites.append(("level_trajectory", check_trajectory(traj)))

    all_pass = True
    print("f3p_attitude verification")
    print("=" * 50)
    for suite_name, checks in suites:
        print(f"\n[{suite_name}]")
        for c in checks:
            status = "PASS" if c.passed else "FAIL"
            print(f"  {status}  {c.name}: {c.detail}")
            all_pass = all_pass and c.passed

    print("\n" + "=" * 50)
    print("OVERALL:", "PASS" if all_pass else "FAIL")
    return all_pass


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_all() else 1)
