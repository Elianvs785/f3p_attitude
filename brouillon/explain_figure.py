"""Visualisation PEDAGOGIQUE d'une figure composee.

Reprend l'esprit des graphes de visualize.py du superviseur :
  - trajectoire 3D avec les reperes (N/E/Up en gris, corps en couleurs chaudes)
  - poussee avec la ligne mg
  - angles (alpha, beta, mu, chi, psi) + residu
et ajoute ce qui manque pour COMPRENDRE la composition :
  - une frise des primitives avec leur part du temps total (le "40% / 60%")
  - une explication texte etape par etape

    python explain_figure.py                 # figure de demonstration 40/60
    python explain_figure.py --learned       # une figure apprise par le RL
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from compose import initial_state, sequence, mix, to_solver_input

COLORS = {"straight": "#4A4A4A", "turn": "#00A79F", "climb": "#E8833A",
          "knife_edge": "#C1272D", "roll": "#7A5AF8"}
EXPLAIN = {
    "straight": "vol rectiligne : gamma, chi et mu sont maintenus",
    "turn": "virage : le cap chi evolue en douceur (smoothstep)",
    "climb": "montee/descente : la pente gamma evolue (bornee a +/-90 deg)",
    "knife_edge": "vol tranche : mu monte a 90 deg, tient, puis revient",
    "roll": "tonneau : mu tourne d'un tour complet autour de la vitesse",
}

try:
    from f3p_attitude.solver import solve_trajectory
    from f3p_attitude.constants import THRUST_MAX, MASS, G
    HAS = True
except ImportError:
    HAS = False
    THRUST_MAX, MASS, G = 1.96, 0.100, 9.81


def positions(seq):
    """Integration trapezoidale, z vers le HAUT (comme integrate_position)."""
    t, g, c, s = seq["t"], seq["gamma"], seq["chi"], seq["speed"]
    v = np.stack([s*np.cos(g)*np.cos(c), s*np.cos(g)*np.sin(c), s*np.sin(g)], 1)
    p = np.zeros_like(v)
    for i in range(len(t)-1):
        p[i+1] = p[i] + 0.5*(v[i]+v[i+1])*(t[i+1]-t[i])
    return p, v


def body_frame(seq, i):
    """Repere du corps approche : x = direction de vol, roule de mu.

    (Avec le solveur, r_wb donne le repere exact ; ici on le reconstruit pour
    que la figure reste lisible meme sans f3p_attitude.)
    """
    g, c, m = seq["gamma"][i], seq["chi"][i], seq["mu"][i]
    xb = np.array([np.cos(g)*np.cos(c), np.cos(g)*np.sin(c), np.sin(g)])
    ref = np.array([0, 0, 1.0]) if abs(xb[2]) < 0.95 else np.array([1.0, 0, 0])
    yb = np.cross(ref, xb); yb /= np.linalg.norm(yb) + 1e-9
    zb = np.cross(xb, yb)
    yb2 = np.cos(m)*yb + np.sin(m)*zb          # roulis autour de la vitesse
    zb2 = -np.sin(m)*yb + np.cos(m)*zb
    return xb, yb2, zb2


def segments(seq):
    out = []
    for (a, b, name) in seq["marks"]:
        i0 = int(np.searchsorted(seq["t"], a))
        i1 = min(int(np.searchsorted(seq["t"], b)) + 1, len(seq["t"]))
        out.append((i0, i1, name, a, b))
    return out


def solve(seq):
    if not HAS:
        return None
    try:
        return solve_trajectory(**to_solver_input(seq))
    except Exception:
        return None


def explain(seq, titre, path, n_frames=6):
    t = seq["t"]
    P, V = positions(seq)
    segs = segments(seq)
    total = float(t[-1])
    res = solve(seq)

    fig = plt.figure(figsize=(17, 9.5))

    # ---------------- 3D avec reperes ----------------
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    for i0, i1, name, a, b in segs:
        ax.plot(P[i0:i1, 0], P[i0:i1, 1], P[i0:i1, 2], color=COLORS[name], lw=2.4)
    rng_xyz = np.ptp(P, axis=0)
    L = max(rng_xyz.max(), 1.0) * 0.055
    for i in np.linspace(0, len(t)-1, n_frames).astype(int):
        xb, yb, zb = body_frame(seq, i)
        for vec, col in [(xb, "#D62728"), (yb, "#FF9896"), (zb, "#9467BD")]:
            ax.quiver(P[i,0], P[i,1], P[i,2], vec[0]*L, vec[1]*L, vec[2]*L,
                      color=col, lw=1.4, arrow_length_ratio=0.25)
    ax.scatter(*P[0], c="green", s=55)
    ax.scatter(*P[-1], c="magenta", s=55)
    # repere visuellement correct : meme echelle sur les 3 axes
    span = max(rng_xyz.max(), 1.0) * 0.6
    ctr = (P.max(0) + P.min(0)) / 2
    ax.set_xlim(ctr[0]-span, ctr[0]+span)
    ax.set_ylim(ctr[1]-span, ctr[1]+span)
    ax.set_zlim(ctr[2]-span, ctr[2]+span)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("North [m]"); ax.set_ylabel("East [m]"); ax.set_zlabel("Up [m]")
    ax.set_title("trajectoire 3D + reperes du corps\n(rouge = nez, rose = aile, violet = dessus)",
                 fontsize=9)

    # ---------------- poussee ----------------
    ax = fig.add_subplot(2, 3, 2)
    if res is not None:
        th = np.asarray(res.thrust, float)
        ax.plot(t, th, color="magenta", lw=1.8, label="T [N]")
        ax.axhline(MASS*G, color="grey", ls=":", label="mg")
        ax.axhline(THRUST_MAX, color="red", ls="--", lw=1, label="T max")
        ax.set_ylim(0, THRUST_MAX*1.1)
    else:
        ax.text(0.5, 0.5, "f3p_attitude absent\n(poussee non calculee)",
                ha="center", va="center", fontsize=10, color="#555555")
    for i0, i1, name, a, b in segs[:-1]:
        ax.axvline(b, color="grey", lw=.5, alpha=.6)
    ax.set_title("Poussee", fontsize=10); ax.set_xlabel("t [s]")
    ax.set_ylabel("Thrust [N]"); ax.grid(alpha=.3); ax.legend(fontsize=7)

    # ---------------- angles + residu ----------------
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(t, np.degrees(seq["mu"]), "--", color="green", lw=1.4, label="mu cmd [deg]")
    ax.plot(t, np.degrees(seq["chi"]), color="#17BECF", lw=1.4, label="chi (cap) [deg]")
    ax.plot(t, np.degrees(seq["gamma"]), color="#1F77B4", lw=1.4, label="gamma [deg]")
    if res is not None:
        ax.plot(t, np.degrees(res.alpha), color="#1F77B4", lw=1.0, alpha=.7, label="alpha [deg]")
        ax.plot(t, np.degrees(res.beta), color="#FF7F0E", lw=1.0, alpha=.7, label="beta [deg]")
        ax2 = ax.twinx()
        ax2.plot(t, np.asarray(res.residual_norm, float), ":", color="red", lw=1.2)
        ax2.set_ylabel("||r||", color="red"); ax2.tick_params(labelcolor="red")
    for i0, i1, name, a, b in segs[:-1]:
        ax.axvline(b, color="grey", lw=.5, alpha=.6)
    ax.set_title("Angles + residu", fontsize=10); ax.set_xlabel("t [s]")
    ax.set_ylabel("Angle [deg]"); ax.grid(alpha=.3); ax.legend(fontsize=6, loc="upper left")

    # ---------------- FRISE de composition (le "40 / 60") ----------------
    ax = fig.add_subplot(2, 1, 2)
    ax.set_position([0.06, 0.05, 0.88, 0.36])
    ax.set_xlim(0, total); ax.set_ylim(0, 1); ax.set_yticks([])
    for i0, i1, name, a, b in segs:
        pct = 100*(b-a)/total
        ax.barh(0.80, b-a, left=a, height=0.22, color=COLORS[name],
                edgecolor="white", lw=1.5)
        if (b-a)/total > 0.06:
            ax.text((a+b)/2, 0.80, f"{name}\n{b-a:.2f}s  ({pct:.0f}%)",
                    ha="center", va="center", fontsize=8,
                    color="white" if name != "straight" else "white")
    # ligne du temps + explication de chaque etape
    txt = []
    for k, (i0, i1, name, a, b) in enumerate(segs, 1):
        pct = 100*(b-a)/total
        d_chi = np.degrees(seq["chi"][i1-1]-seq["chi"][i0])
        d_gam = np.degrees(seq["gamma"][i1-1]-seq["gamma"][i0])
        d_mu  = np.degrees(seq["mu"][i1-1]-seq["mu"][i0])
        chg = []
        if abs(d_chi) > 1: chg.append(f"cap {d_chi:+.0f} deg")
        if abs(d_gam) > 1: chg.append(f"pente {d_gam:+.0f} deg")
        if abs(d_mu) > 1:  chg.append(f"roulis {d_mu:+.0f} deg")
        chg = ", ".join(chg) if chg else "etat maintenu"
        txt.append(f"{k}. {name:<11s} {b-a:.2f}s = {pct:4.0f}% du temps  |  {chg}"
                   f"\n     {EXPLAIN[name]}")
    ax.text(0.0, 0.60, "\n".join(txt), transform=ax.transAxes, va="top",
            family="monospace", fontsize=8.5)
    ax.set_xlabel("t [s]")
    ax.set_title("Composition : quelle primitive, pendant quelle part du temps, et ce qu'elle change",
                 fontsize=10)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)

    fig.suptitle(titre, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    # --- explication texte dans la console ---
    print(f"\n{titre}")
    print("-"*len(titre))
    for k, (i0, i1, name, a, b) in enumerate(segs, 1):
        pct = 100*(b-a)/total
        print(f"  {k}. {name:<11s} {a:5.2f} -> {b:5.2f}s  ({pct:4.0f}% du temps)  "
              f"{EXPLAIN[name]}")
    if res is not None:
        th = np.asarray(res.thrust, float)
        print(f"  -> poussee {th.max():.2f}/{THRUST_MAX:.2f} N ({100*th.max()/THRUST_MAX:.0f}%), "
              f"residu max {np.asarray(res.residual_norm).max():.1e}")
    return path


if __name__ == "__main__":
    st = initial_state(speed=5.0)

    if "--learned" in sys.argv:
        import pickle
        from figure_rl import rollout, CATALOG
        pol = pickle.load(open("policy_figures.pkl", "rb"))
        from mock_solver import mock
        solver_fn = (lambda seq: (float(np.asarray(solve(seq).thrust).max()/THRUST_MAX),
                                  float(np.degrees(np.abs(np.asarray(solve(seq).alpha))).max()),
                                  float(np.asarray(solve(seq).residual_norm).max()))) if HAS else mock
        rng = np.random.default_rng(0)
        best = None
        for _ in range(30):
            R, _, mv, sq, info = rollout(pol, rng, 6, solver_fn)
            if info["feasible"] and (best is None or R > best[0]):
                sq["moves"] = mv; best = (R, sq)
        explain(best[1], f"Figure apprise par le RL — reward {best[0]:+.3f}",
                "explain_learned.png")
    else:
        # 1) la demonstration du "40% droit / 60% virage"
        seq = mix(st, 2.5, [("straight", 0.4), ("turn", 0.6)],
                  {"turn": {"delta_chi_deg": 90.0}})
        explain(seq, "Composition 40% ligne droite / 60% virage (90 deg)",
                "explain_40_60.png")
        # 2) une figure complete
        seq2 = sequence(st, [("straight", 0.8, {}),
                             ("climb", 1.2, {"delta_gamma_deg": 45.0}),
                             ("roll", 1.5, {"n_turns": 1.0}),
                             ("turn", 1.2, {"delta_chi_deg": 180.0}),
                             ("climb", 1.2, {"delta_gamma_deg": -45.0}),
                             ("knife_edge", 1.6, {"mu_deg": 90.0})])
        explain(seq2, "Figure composee de 6 primitives", "explain_figure.png")
