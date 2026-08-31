"""Visualisation d'une figure composee par MELANGE CONTINU.

Le panneau central est le plus important : un graphe empile des poids w_i(t).
On y lit directement "a cet instant, c'est 77 % virage et 23 % ligne droite".
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from blend import blend, mix_at, PRIMS, N_PRIMS

COL = {"straight":"#4A4A4A","turn_right":"#00A79F","turn_left":"#00D4C8",
       "climb_up":"#E8833A","climb_down":"#F5B98A","roll_right":"#7A5AF8",
       "roll_left":"#B3A0FA","accel":"#2E8B57","decel":"#8B2E2E"}

try:
    from f3p_attitude.solver import solve_trajectory
    from f3p_attitude.constants import THRUST_MAX, MASS, G
    HAS = True
except ImportError:
    HAS = False; THRUST_MAX, MASS, G = 1.96, 0.100, 9.81


def positions(seq):
    t,g,c,s = seq["t"],seq["gamma"],seq["chi"],seq["speed"]
    v = np.stack([s*np.cos(g)*np.cos(c), s*np.cos(g)*np.sin(c), s*np.sin(g)],1)
    p = np.zeros_like(v)
    for i in range(len(t)-1):
        p[i+1] = p[i] + 0.5*(v[i]+v[i+1])*(t[i+1]-t[i])
    return p


def solve(seq):
    if not HAS: return None
    try:
        return solve_trajectory(seq["t"], seq["gamma"], seq["chi"], seq["speed"],
                                seq["mu"], gamma_dot=seq["gamma_dot"],
                                chi_dot=seq["chi_dot"], speed_dot=seq["speed_dot"])
    except Exception:
        return None


def plot(seq, titre, path, marks=(0.2, 0.4, 0.6, 0.8)):
    t = seq["t"]; W = seq["weights"]; P = positions(seq)
    res = solve(seq)
    fig = plt.figure(figsize=(17, 9))

    # --- 3D ---
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.plot(P[:,0], P[:,1], P[:,2], color="#00A79F", lw=2.2)
    ax.scatter(*P[0], c="green", s=50); ax.scatter(*P[-1], c="magenta", s=50)
    r = np.ptp(P,axis=0); span = max(r.max(),1.0)*0.6; ctr=(P.max(0)+P.min(0))/2
    ax.set_xlim(ctr[0]-span,ctr[0]+span); ax.set_ylim(ctr[1]-span,ctr[1]+span)
    ax.set_zlim(ctr[2]-span,ctr[2]+span); ax.set_box_aspect((1,1,1))
    ax.set_xlabel("North [m]"); ax.set_ylabel("East [m]"); ax.set_zlabel("Up [m]")
    ax.set_title("trajectoire 3D", fontsize=10)

    # --- poussee ---
    ax = fig.add_subplot(2, 3, 2)
    if res is not None:
        th = np.asarray(res.thrust,float)
        ax.plot(t, th, color="magenta", lw=1.8, label="T [N]")
        ax.axhline(MASS*G, color="grey", ls=":", label="mg")
        ax.axhline(THRUST_MAX, color="red", ls="--", lw=1, label="T max")
        ax.set_ylim(0, THRUST_MAX*1.1); ax.legend(fontsize=7)
    else:
        ax.text(.5,.5,"f3p_attitude absent",ha="center",va="center",color="#555")
    ax.set_title("Poussee",fontsize=10); ax.set_xlabel("t [s]"); ax.grid(alpha=.3)

    # --- angles + residu ---
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(t, np.degrees(seq["gamma"]), label="gamma", lw=1.4)
    ax.plot(t, np.degrees(seq["chi"]), label="chi", lw=1.4)
    ax.plot(t, np.degrees(seq["mu"]), "--", label="mu", lw=1.4)
    if res is not None:
        ax.plot(t, np.degrees(res.alpha), lw=1.0, alpha=.7, label="alpha")
        ax2 = ax.twinx()
        ax2.plot(t, np.asarray(res.residual_norm,float), ":", color="red", lw=1.2)
        ax2.set_ylabel("||r||", color="red"); ax2.tick_params(labelcolor="red")
    ax.set_title("Angles + residu",fontsize=10); ax.set_xlabel("t [s]")
    ax.set_ylabel("deg"); ax.grid(alpha=.3); ax.legend(fontsize=7)

    # --- LE PANNEAU CLE : melange des primitives dans le temps ---
    ax = fig.add_subplot(2, 1, 2)
    used = [i for i in range(N_PRIMS) if W[:, i].max() > 0.01]
    ax.stackplot(t, *[W[:, i] for i in used],
                 labels=[PRIMS[i] for i in used],
                 colors=[COL[PRIMS[i]] for i in used], alpha=.92)
    ax.set_ylim(0, 1); ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("part de chaque primitive"); ax.set_xlabel("t [s]")
    ax.set_title("MELANGE CONTINU : a chaque instant, le mouvement est une combinaison ponderee",
                 fontsize=11, pad=34)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), fontsize=8, ncol=len(used), frameon=False)
    # annotations : le melange exact a quelques instants
    for frac in marks:
        ts = t[0] + frac*(t[-1]-t[0])
        m = mix_at(seq, ts, top=2)
        lbl = " + ".join(f"{100*w:.0f}% {n}" for n, w in m)
        ax.axvline(ts, color="black", lw=1.0, alpha=.7)
        ax.text(ts, 1.015, f"t={ts:.1f}s\n{lbl}", ha="center", va="bottom",
                fontsize=8, family="monospace")

    fig.suptitle(titre, fontsize=13)
    plt.tight_layout(rect=[0,0.04,1,0.95])
    plt.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {path}")
    for frac in marks:
        ts = t[0] + frac*(t[-1]-t[0])
        m = mix_at(seq, ts, top=3)
        print(f"     t={ts:4.1f}s : " + " + ".join(f"{100*w:.0f}% {n}" for n,w in m))
    return path


if __name__ == "__main__":
    i = {p: PRIMS.index(p) for p in PRIMS}
    def w(**kw):
        v = np.zeros(N_PRIMS)
        for k, x in kw.items(): v[i[k]] = x
        return v

    print("1) virage progressif (straight -> turn -> straight)")
    K = np.array([w(straight=1), w(straight=.3, turn_right=.7), w(turn_right=1),
                  w(straight=.5, turn_right=.5), w(straight=1)])
    plot(blend(K, 3.0), "Virage progressif par melange continu", "blend_turn.png")

    print("\n2) figure combinee : virage + montee + tonneau simultanes")
    K2 = np.array([w(straight=1),
                   w(turn_right=.6, climb_up=.4),
                   w(turn_right=.3, climb_up=.3, roll_right=.4),
                   w(roll_right=.7, straight=.3),
                   w(turn_left=.5, climb_down=.5),
                   w(straight=1)])
    plot(blend(K2, 5.0), "Figure combinee (primitives simultanees)", "blend_figure.png")
