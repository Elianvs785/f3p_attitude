"""Example (gamma, chi, V, mu) paths for verify and visualize."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from f3p_attitude.kinematics import velocity_world
from f3p_attitude.solver import TrajectoryResult, solve_trajectory


@dataclass
class FlightScenario:
    name: str
    title: str
    t: np.ndarray
    gamma: np.ndarray
    chi: np.ndarray
    speed: np.ndarray
    mu: np.ndarray
    gamma_dot: np.ndarray | None = None
    chi_dot: np.ndarray | None = None
    speed_dot: np.ndarray | None = None


def integrate_position(t: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    """Trapezoidal integration; p(0)=0."""
    n = len(t)
    p = np.zeros((n, 3))
    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        p[i + 1] = p[i] + 0.5 * (v_world[i] + v_world[i + 1]) * dt
    return p


def solve_scenario(sc: FlightScenario) -> tuple[TrajectoryResult, np.ndarray]:
    traj = solve_trajectory(
        sc.t,
        sc.gamma,
        sc.chi,
        sc.speed,
        sc.mu,
        gamma_dot=sc.gamma_dot,
        chi_dot=sc.chi_dot,
        speed_dot=sc.speed_dot,
    )
    v_world = np.array(
        [velocity_world(sc.gamma[i], sc.chi[i], sc.speed[i])[0] for i in range(len(sc.t))]
    )
    return traj, integrate_position(sc.t, v_world)


def smoothstep01(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def level_to_vertical_gamma(
    t: np.ndarray,
    level_duration: float,
    blend_duration: float,
    gamma_vertical: float = np.deg2rad(90.0),
) -> tuple[np.ndarray, np.ndarray]:
    gamma = np.zeros_like(t, dtype=float)
    gamma_dot = np.zeros_like(t, dtype=float)
    t0 = level_duration
    t1 = level_duration + blend_duration
    g_vert = gamma_vertical
    for i, ti in enumerate(t):
        if ti <= t0:
            gamma[i] = 0.0
            gamma_dot[i] = 0.0
        elif ti >= t1:
            gamma[i] = g_vert
            gamma_dot[i] = 0.0
        else:
            u = (ti - t0) / blend_duration
            s = smoothstep01(np.array([u]))[0]
            gamma[i] = g_vert * s
            gamma_dot[i] = g_vert * 6.0 * u * (1.0 - u) / blend_duration
    return gamma, gamma_dot


def level_to_vertical_profile(
    t: np.ndarray,
    *,
    level_duration: float = 4.0,
    blend_duration: float = 4.0,
    vertical_hold: float = 4.0,
    speed: float = 5.0,
    chi: float = 0.0,
    mu: float = 0.0,
    gamma_vertical: float = np.deg2rad(90.0),
) -> dict[str, np.ndarray]:
    gamma, gamma_dot = level_to_vertical_gamma(
        t, level_duration, blend_duration, gamma_vertical
    )
    n = len(t)
    return {
        "t": t,
        "gamma": gamma,
        "gamma_dot": gamma_dot,
        "chi": np.full(n, chi),
        "chi_dot": np.zeros(n),
        "speed": np.full(n, speed),
        "speed_dot": np.zeros(n),
        "mu": np.full(n, mu),
        "vertical_start": np.array([level_duration + blend_duration]),
    }


def scenario_level_constant_speed(
    speed: float,
    duration: float = 4.0,
    dt: float = 0.1,
    mu: float = 0.0,
    name: str | None = None,
    title: str | None = None,
) -> FlightScenario:
    t = np.arange(0.0, duration + dt * 0.5, dt)
    tag = f"{speed:.2f}".replace(".", "p")
    return FlightScenario(
        name=name or f"level_V{tag}",
        title=title or f"Level flight V={speed:.2f} m/s, mu=0",
        t=t,
        gamma=np.zeros_like(t),
        chi=np.zeros_like(t),
        speed=np.full_like(t, speed),
        mu=np.full_like(t, mu),
    )


def scenario_level_coordinated(duration: float = 4.0, dt: float = 0.1) -> FlightScenario:
    sc = scenario_level_constant_speed(5.0, duration, dt)
    return FlightScenario(
        name="01_level_mu0",
        title="Level coordinated (mu=0)",
        t=sc.t,
        gamma=sc.gamma,
        chi=sc.chi,
        speed=sc.speed,
        mu=sc.mu,
    )


def scenario_knife_edge(duration: float = 4.0, dt: float = 0.1) -> FlightScenario:
    t = np.arange(0.0, duration + dt * 0.5, dt)
    return FlightScenario(
        name="02_knife_mu90",
        title="Knife edge (mu=90 deg)",
        t=t,
        gamma=np.zeros_like(t),
        chi=np.zeros_like(t),
        speed=np.full_like(t, 5.0),
        mu=np.full_like(t, np.deg2rad(90.0)),
    )


def scenario_level_to_vertical(
    duration: float = 12.0,
    dt: float = 0.1,
    level_duration: float = 4.0,
    blend_duration: float = 4.0,
    speed: float = 5.0,
) -> FlightScenario:
    t = np.arange(0.0, duration + dt * 0.5, dt)
    profile = level_to_vertical_profile(
        t,
        level_duration=level_duration,
        blend_duration=blend_duration,
        vertical_hold=duration - level_duration - blend_duration,
        speed=speed,
        mu=0.0,
    )
    return FlightScenario(
        name="03_level_to_vertical",
        title="Level -> vertical (smooth pitch-up)",
        t=t,
        gamma=profile["gamma"],
        chi=profile["chi"],
        speed=profile["speed"],
        mu=profile["mu"],
        gamma_dot=profile["gamma_dot"],
        chi_dot=profile["chi_dot"],
        speed_dot=profile["speed_dot"],
    )


def all_scenarios() -> list[FlightScenario]:
    """Default figures: same cases as core verify checks."""
    return [
        scenario_level_coordinated(),
        scenario_knife_edge(),
        scenario_level_to_vertical(),
    ]
