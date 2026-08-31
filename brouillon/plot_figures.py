"""Visualisation des figures apprises par le RL.

S'inspire de visualize.py du superviseur (plot_scenario_3d / plot_scenario_views)
et reprend son integrate_position pour reconstruire la trajectoire a partir de
(gamma, chi, speed).

    python plot_figures.py            # entraine puis trace
    python plot_figures.py --load     # recharge une politique sauvegardee
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figure_rl import train, rollout, CATALOG, N_MOVES
from compose import to_solver_input

COLORS = {"straight": "#4A4A4A", "turn": "#00A79F", "climb": "#E8833A",
          "knife_edge": "#C1272D", "roll": "#7A5AF8"}

try:
    from f3p_attitude.solver import solve_trajectory
    from f3p_attitude.constants import THRUST_MAX
    HAS_SOLVER = True
    def solver(seq):
        try:
            r = solve_trajectory(**to_solver_input(seq))
        except Exception:
            return None
        return (float(np.asarray(r.thrust).max()/THRUST_MAX),
                float(np.degrees(np.abs(np.asarray(r.alpha))).max()),
                float(np.asarray(r.residual_norm).max()))
except ImportError:
    from mock_solver import mock as solver
    THRUST_MAX = 1.0
    HAS_SOLVER = False


def integrate_position(t, gamma, chi, speed):
    """Reprend integrate_position de scenarios.py (integration trapezoidale),
    avec z vers le HAUT pour que l'altitude se lise naturellement."""
    v = np.stack([speed*np.cos(gamma)*np.cos(chi),
                  speed*np.cos(gamma)*np.sin(chi),
                  speed*np.sin(gamma)], axis=1)
    p = np.zeros_like(v)
    for i in range(len(t)-1):
        p[i+1] = p[i] + 0.5*(v[i]+v[i+1])*(t[i+1]-t[i])
    return p


def segment_bounds(seq):
    """Indices de debut/fin de chaque primitive, avec son nom."""
    out = []
    for (a, b, name) in seq["marks"]:
        i0 = int(np.searchsorted(seq["t"], a))
        i1 = int(np.searchsorted(seq["t"], b))
        out.append((i0, min(i1+1, len(seq["t"])), name))
    return out


def plot_figure(seq, info, moves, titre, path):
    t = seq["t"]
    P = integrate_position(t, seq["gamma"], seq["chi"], seq["speed"])
    segs = segment_bounds(seq)

    fig = plt.figure(figsize=(16, 9))

    # --- 3D, colore par primitive ---
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    for i0, i1, name in segs:
        ax.plot(P[i0:i1,0], P[i0:i1,1], P[i0:i1,2],
                color=COLORS[name], lw=2.2)
    ax.scatter(*P[0], c="green", s=50)
    ax.scatter(*P[-1], c="magenta", s=50)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("altitude (m)")
    ax.set_title("trajectoire 3D", fontsize=10)
    ax.legend(handles=[Line2D([0],[0], color=c, lw=2.4, label=n)
                       for n, c in COLORS.items()], fontsize=7, loc="upper left")

    # --- vue de dessus / de cote ---
    for k, (ix, iy, lab) in enumerate([(0,1,"vue de dessus (x, y)"),
                                       (0,2,"vue de cote (x, altitude)")]):
        a = fig.add_subplot(2, 3, k+2)
        for i0, i1, name in segs:
            a.plot(P[i0:i1,ix], P[i0:i1,iy], color=COLORS[name], lw=2.0)
        a.scatter(P[0,ix], P[0,iy], c="green", s=40)
        a.scatter(P[-1,ix], P[-1,iy], c="magenta", s=40)
        a.set_title(lab, fontsize=10); a.grid(alpha=.3); a.set_aspect("equal")

    # --- profils gamma / chi / mu ---
    a = fig.add_subplot(2, 3, 4)
    for key, lab in [("gamma","gamma (pente)"), ("chi","chi (cap)"), ("mu","mu (roulis)")]:
        a.plot(t, np.degrees(seq[key]), lw=1.6, label=lab)
    for i0, i1, name in segs[:-1]:
        a.axvline(t[i1-1], color="grey", lw=.5, alpha=.6)
    a.set_title("profils (deg)", fontsize=10); a.legend(fontsize=7)
    a.grid(alpha=.3); a.set_xlabel("t (s)")

    # --- vitesse ---
    a = fig.add_subplot(2, 3, 5)
    a.plot(t, seq["speed"], color="#00A79F", lw=1.8)
    a.axhline(3.0, color="red", ls="--", lw=1)
    a.axhline(7.0, color="red", ls="--", lw=1)
    for i0, i1, name in segs[:-1]:
        a.axvline(t[i1-1], color="grey", lw=.5, alpha=.6)
    a.set_title("vitesse (m/s) — limites 3-7", fontsize=10)
    a.grid(alpha=.3); a.set_xlabel("t (s)"); a.set_ylim(0, 8)

    # --- composition et indicateurs ---
    a = fig.add_subplot(2, 3, 6); a.axis("off")
    txt = "COMPOSITION\n\n"
    for i, (idx, dur) in enumerate(moves, 1):
        n, kw, sp = CATALOG[idx]
        p = ", ".join(f"{k}={v}" for k, v in kw.items()) or "-"
        txt += f" {i}. {n:<11s} {dur:.1f}s  {p}\n"
    txt += f"\nduree totale : {t[-1]:.1f} s\n"
    txt += f"\nINDICATEURS\n"
    txt += f"  poussee        {100*info['thrust']:.0f} %  (cible 85)\n"
    txt += f"  angle attaque  {info['alpha']:.0f} deg (decrochage 25)\n"
    txt += f"  residu         {info['res']:.1e}\n"
    txt += f"  roulis max     {info['roll']:.0f} deg/s\n"
    txt += f"\nSCORES\n"
    txt += f"  diversite      {info['s_div']:.2f}\n"
    txt += f"  non-repetition {info['s_rep']:.2f}\n"
    txt += f"  amplitude      {info['s_amp']:.2f}\n"
    a.text(0.0, 0.98, txt, va="top", family="monospace", fontsize=9)

    fig.suptitle(titre, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("solveur :", "f3p_attitude (reel)" if HAS_SOLVER else "SIMULE")
    if "--load" in sys.argv and os.path.exists("policy_figures.pkl"):
        pol = pickle.load(open("policy_figures.pkl", "rb"))
        print("politique rechargee")
    else:
        print("entrainement...")
        pol, hist, feas = train(solver, episodes=400, n_steps=6, lr=0.05,
                                seed=2, log_every=100)
        pickle.dump(pol, open("policy_figures.pkl", "wb"))

    rng = np.random.default_rng(0)
    best = []
    for _ in range(40):
        R, _, mv, sq, info = rollout(pol, rng, 6, solver)
        if info["feasible"]:
            sq["moves"] = mv
            best.append((R, mv, sq, info))
    best.sort(key=lambda x: -x[0])
    print(f"\n{len(best)} figures volables sur 40 tirages")
    for rank, (R, mv, sq, info) in enumerate(best[:3], 1):
        path = f"figure_{rank}.png"
        plot_figure(sq, info, mv, f"Figure apprise #{rank} — reward {R:+.3f}", path)
        print(f"  #{rank} reward {R:+.3f} | poussee {100*info['thrust']:.0f}% -> {path}")