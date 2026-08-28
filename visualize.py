"""
Plot solver results for example scenarios.

Run:
    python -m f3p_attitude.visualize
    python -m f3p_attitude.visualize --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from f3p_attitude.constants import MASS
from f3p_attitude.frames import velocity_unit_ned
from f3p_attitude.kinematics import velocity_world
from f3p_attitude.ned_display import gravity_display, pos_ned_to_display, vec_ned_to_display
from f3p_attitude.scenarios import (
    FlightScenario,
    all_scenarios,
    scenario_level_constant_speed,
    solve_scenario,
)
from f3p_attitude.solver import TrajectoryResult


def triad_ned_to_display(basis_rows: np.ndarray) -> np.ndarray:
    r = np.asarray(basis_rows, dtype=float)
    s = np.diag([1.0, 1.0, -1.0])
    return s @ r @ s


def wind_basis_world(gamma: float, chi: float) -> np.ndarray:
    x_w = velocity_unit_ned(gamma, chi)
    y_w = np.array([-np.sin(chi), np.cos(chi), 0.0])
    z_w = np.cross(x_w, y_w)
    return np.vstack([x_w, y_w, z_w])


def draw_triad(ax, origin, basis_rows, scale, labels, colors, linewidth=2.0, alpha=1.0):
    for i, (lab, col) in enumerate(zip(labels, colors)):
        end = origin + scale * basis_rows[i]
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            color=col,
            linewidth=linewidth,
            alpha=alpha,
            label=lab if alpha == 1.0 else None,
        )


def draw_ned_reference(ax, origin: np.ndarray, scale: float) -> None:
    axes = np.eye(3)
    labels = ("N (x)", "E (y)", "D (z)")
    for i in range(3):
        end = origin + scale * axes[i]
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            linestyle="--",
            color="0.35",
            linewidth=1.2,
            alpha=0.55,
        )
        ax.text(end[0], end[1], end[2], labels[i], fontsize=7, color="0.4")


def draw_frames_on_path_3d(
    ax3d,
    pos_disp: np.ndarray,
    traj: TrajectoryResult,
    gamma: np.ndarray,
    chi: np.ndarray,
    speed: np.ndarray,
    sample_indices: list[int] | None = None,
) -> None:
    n = len(pos_disp)
    if sample_indices is None:
        sample_indices = [0, n // 2, n - 1]
    span = max(float(np.ptp(pos_disp, axis=0).max()), 1.0)
    frame_scale = 0.35 * span
    vel_scale = 0.25 * span
    alphas = [1.0, 0.75, 0.55]
    if len(sample_indices) != len(alphas):
        alphas = [1.0] + [0.7] * (len(sample_indices) - 2) + [0.55]
        alphas = alphas[: len(sample_indices)]
    first = sample_indices[0]
    for idx, alpha in zip(sample_indices, alphas):
        origin = pos_disp[idx]
        g, c, spd = gamma[idx], chi[idx], speed[idx]
        r_wb = triad_ned_to_display(traj.r_wb[idx])
        r_ww = triad_ned_to_display(wind_basis_world(g, c))
        v_w_d = vec_ned_to_display(velocity_world(g, c, spd)[0])
        if idx == first:
            o_ref = origin - 0.15 * frame_scale * np.array([1.0, 1.0, 0.0])
            draw_ned_reference(ax3d, o_ref, frame_scale)
            draw_triad(
                ax3d, origin, triad_ned_to_display(np.eye(3)), frame_scale,
                ("N", "E", "Up"), ("0.35", "0.35", "0.35"), linewidth=2.0, alpha=alpha,
            )
            draw_triad(
                ax3d, origin, r_ww, frame_scale,
                ("wind x", "wind y", "wind z"), ("#2ca02c", "#98df8a", "#1a6b1a"),
                linewidth=2.0, alpha=alpha,
            )
            draw_triad(
                ax3d, origin, r_wb, frame_scale,
                ("body x (nose)", "body y", "body z"),
                ("#d62728", "#ff7f0e", "#9467bd"), linewidth=2.8, alpha=alpha,
            )
        else:
            draw_triad(
                ax3d, origin, r_wb, frame_scale * 0.85, ("", "", ""),
                ("#d62728", "#ff7f0e", "#9467bd"), linewidth=1.8, alpha=alpha,
            )
        ax3d.quiver(
            origin[0], origin[1], origin[2],
            v_w_d[0], v_w_d[1], v_w_d[2],
            length=vel_scale, normalize=True, color="#17becf", linewidth=1.5, alpha=alpha,
        )
        thrust_dir = vec_ned_to_display(traj.r_wb[idx, 0])
        ax3d.quiver(
            origin[0], origin[1], origin[2],
            thrust_dir[0], thrust_dir[1], thrust_dir[2],
            length=vel_scale * (0.5 + 0.5 * traj.thrust[idx] / (MASS * 9.81)),
            normalize=True, color="#e377c2", linewidth=1.5, alpha=alpha,
        )
    o0 = pos_disp[0]
    g_scale = 0.2 * span
    gw = gravity_display() / np.linalg.norm(gravity_display())
    ax3d.quiver(
        o0[0], o0[1], o0[2], gw[0], gw[1], gw[2],
        length=g_scale, normalize=True, color="0.5", linewidth=1.2,
    )


def _set_axes_equal_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.5 * np.max(maxs - mins), 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=24, azim=-58)


def plot_scenario_3d(
    sc: FlightScenario,
    traj: TrajectoryResult,
    position: np.ndarray,
    output_path: Path | None,
    show: bool,
) -> None:
    fig = plt.figure(figsize=(14, 6))
    ax3d = fig.add_subplot(121, projection="3d")
    ax_ts = fig.add_subplot(222)
    ax_tb = fig.add_subplot(224)
    pos_disp = pos_ned_to_display(position)
    ax3d.plot(pos_disp[:, 0], pos_disp[:, 1], pos_disp[:, 2], "k-", lw=2.0)
    ax3d.scatter(*pos_disp[0], c="green", s=40)
    ax3d.scatter(*pos_disp[-1], c="red", s=40)
    draw_frames_on_path_3d(ax3d, pos_disp, traj, sc.gamma, sc.chi, sc.speed)
    ax3d.set_xlabel("North [m]")
    ax3d.set_ylabel("East [m]")
    ax3d.set_zlabel("Up [m]")
    ax3d.set_title(sc.title)
    _set_axes_equal_3d(ax3d, pos_disp)
    ax_ts.plot(sc.t, traj.thrust, "m-", label="T [N]")
    ax_ts.axhline(MASS * 9.81, color="gray", ls=":")
    ax_ts.set_ylabel("Thrust [N]")
    ax_ts.set_xlabel("t [s]")
    ax_ts.grid(True, alpha=0.3)
    ax_tb.plot(sc.t, np.rad2deg(traj.alpha), label="alpha [deg]")
    ax_tb.plot(sc.t, np.rad2deg(traj.beta), label="beta [deg]")
    ax_tb.plot(sc.t, np.rad2deg(sc.mu), "--", label="mu cmd [deg]")
    ax_tb2 = ax_tb.twinx()
    ax_tb2.plot(sc.t, traj.residual_norm, "r:", alpha=0.7)
    ax_tb2.set_ylabel("||r||")
    ax_tb.set_xlabel("t [s]")
    ax_tb.grid(True, alpha=0.3)
    fig.suptitle(f"{sc.name}: max ||r||={traj.residual_norm.max():.2e}")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  saved {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _set_2d_view_limits(ax, horiz, vert, *, min_half_span=4.0, pad_ratio=0.2):
    cx = 0.5 * (float(horiz.min()) + float(horiz.max()))
    cy = 0.5 * (float(vert.min()) + float(vert.max()))
    half = max(
        max(0.5 * float(np.ptp(horiz)), min_half_span),
        max(0.5 * float(np.ptp(vert)), min_half_span),
    ) * (1.0 + pad_ratio)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")
    return half


def _quiver_dirs(ax, x0, y0, dx, dy, half_span, *, color="#d62728"):
    arrow_len = 0.14 * (2.0 * half_span)
    norm = np.hypot(dx, dy)
    mask = norm > 1e-9
    u = np.zeros_like(dx)
    v = np.zeros_like(dy)
    u[mask] = dx[mask] / norm[mask]
    v[mask] = dy[mask] / norm[mask]
    ax.quiver(
        x0, y0, u, v, angles="xy", scale_units="xy", scale=1.0 / arrow_len,
        color=color, width=0.005, headwidth=4.0, headlength=5.0, alpha=0.85,
    )


def plot_scenario_views(
    sc: FlightScenario,
    traj: TrajectoryResult,
    position: np.ndarray,
    output_path: Path | None,
    show: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    step = max(1, len(sc.t) // 20)
    pos_s = position[::step]
    bx = traj.r_wb[::step, 0, :]
    vel_s = np.array(
        [velocity_world(sc.gamma[i], sc.chi[i], sc.speed[i])[0] for i in range(0, len(sc.t), step)]
    )
    alt = -position[:, 2]
    alt_s = -pos_s[:, 2]
    vel_d = vel_s.copy()
    vel_d[:, 2] *= -1.0
    bx_d = bx.copy()
    bx_d[:, 2] *= -1.0
    min_half = max(4.0, 0.22 * float(np.sum(np.linalg.norm(np.diff(position, axis=0), axis=1))))
    ax = axes[0]
    ax.plot(position[:, 0], position[:, 1], "k-", lw=2)
    half_ne = _set_2d_view_limits(ax, position[:, 0], position[:, 1], min_half_span=min_half)
    _quiver_dirs(ax, pos_s[:, 0], pos_s[:, 1], vel_s[:, 0], vel_s[:, 1], half_ne, color="#17becf")
    _quiver_dirs(ax, pos_s[:, 0], pos_s[:, 1], bx[:, 0], bx[:, 1], half_ne, color="#d62728")
    ax.set_title("Top: cyan=v, red=body +x")
    ax = axes[1]
    ax.plot(position[:, 0], alt, "k-", lw=2)
    half_xu = _set_2d_view_limits(ax, position[:, 0], alt, min_half_span=min_half)
    _quiver_dirs(ax, pos_s[:, 0], alt_s, vel_d[:, 0], vel_d[:, 2], half_xu, color="#17becf")
    _quiver_dirs(ax, pos_s[:, 0], alt_s, bx_d[:, 0], bx_d[:, 2], half_xu, color="#d62728")
    ax.set_ylabel("Up [m]")
    fig.suptitle(sc.title)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  saved {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def low_speed_sweep_speeds(v_min=0.0, v_max=8.0, step=0.5) -> tuple[float, ...]:
    return tuple(np.arange(v_min, v_max + 0.25 * step, step))


def plot_low_speed_level_sweep(
    output_path: Path | None,
    show: bool,
    speeds: tuple[float, ...] | None = None,
) -> None:
    if speeds is None:
        speeds = low_speed_sweep_speeds()
    v_arr = np.asarray(speeds, dtype=float)
    thrust = np.zeros_like(v_arr)
    alpha_deg = np.zeros_like(v_arr)
    mg = MASS * 9.81
    print("\n[low_speed_sweep] gamma=0, mu=0")
    for i, spd in enumerate(speeds):
        traj, _ = solve_scenario(scenario_level_constant_speed(spd))
        thrust[i] = traj.thrust[0]
        alpha_deg[i] = np.rad2deg(traj.alpha[0])
    fig, ax_t = plt.subplots(figsize=(9, 5))
    ax_t.plot(v_arr, thrust, "m-o", lw=2, label="Thrust T")
    ax_t.axhline(mg, color="gray", ls=":", label="mg")
    ax_t.set_xlabel("Commanded speed V [m/s]")
    ax_t.set_ylabel("Thrust T [N]", color="m")
    ax_a = ax_t.twinx()
    ax_a.plot(v_arr, alpha_deg, "b-s", lw=2, label=r"$\alpha$")
    ax_a.set_ylabel(r"$\alpha$ [deg]", color="b")
    lines_t, labels_t = ax_t.get_legend_handles_labels()
    lines_a, labels_a = ax_a.get_legend_handles_labels()
    ax_t.legend(lines_t + lines_a, labels_t + labels_a, loc="center right", fontsize=9)
    fig.suptitle("Low-speed level sweep")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  saved {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def run_visual(output_dir: Path | str = "f3p_attitude/output", show: bool = False) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("f3p_attitude visualize")
    print(f"Output: {out.resolve()}")
    for sc in all_scenarios():
        print(f"\n[{sc.name}] {sc.title}")
        traj, pos = solve_scenario(sc)
        plot_scenario_3d(sc, traj, pos, out / f"{sc.name}_3d.png", show)
        plot_scenario_views(sc, traj, pos, out / f"{sc.name}_views.png", show)
    plot_low_speed_level_sweep(out / "low_speed_level_sweep.png", show)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot example scenarios")
    parser.add_argument("--output", type=str, default="f3p_attitude/output")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run_visual(output_dir=args.output, show=args.show)


if __name__ == "__main__":
    main()
