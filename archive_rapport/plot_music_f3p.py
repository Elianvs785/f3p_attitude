"""Visualisation de la choregraphie F3P pilotee par la musique.

    python plot_music_f3p.py            # utilise beat_times.npy si present
    python plot_music_f3p.py 30         # limite a 30 beats
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from music_f3p import load_beats, build, positions
from blend import PRIMS, N_PRIMS

COL = {"straight":"#4A4A4A","turn_right":"#00A79F","turn_left":"#7FD8D2",
       "climb_up":"#E8833A","climb_down":"#F5C6A0","roll_right":"#7A5AF8",
       "roll_left":"#B9A8FB","accel":"#2E8B57","decel":"#A03030"}
NIV_COL = {"doux":"#9ECAE1","moyen":"#F5B041","fort":"#C0392B"}

try:
    from f3p_attitude.solver import solve_trajectory
    from f3p_attitude.constants import THRUST_MAX, MASS, G, RESIDUAL_TOL
    HAS = True
except ImportError:
    HAS = False; THRUST_MAX, MASS, G, RESIDUAL_TOL = 1.96, .1, 9.81, 1e-4


def main(max_beats=None):
    t, f, src = load_beats()
    rng = np.random.default_rng(1)
    seq = build(t, f, rng, max_beats=max_beats)
    P = positions(seq); tt = seq["t"]; W = seq["weights"]
    beats = seq["beat_times"][:-1]
    forces = np.array([c[1] for c in seq["choix"]])
    niveaux = [c[2] for c in seq["choix"]]

    res = None
    if HAS:
        try:
            res = solve_trajectory(tt, seq["gamma"], seq["chi"], seq["speed"], seq["mu"],
                                   gamma_dot=seq["gamma_dot"], chi_dot=seq["chi_dot"],
                                   speed_dot=seq["speed_dot"])
        except Exception:
            res = None

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, .8, 1.25], hspace=.6, wspace=.3)

    # 3D, points rouges sur les beats
    ax = fig.add_subplot(gs[0, 0:2], projection="3d")
    ax.plot(P[:,0], P[:,1], P[:,2], color="#00A79F", lw=1.8)
    idx = [int(np.clip(np.searchsorted(tt, b), 0, len(tt)-1)) for b in beats]
    ax.scatter(P[idx,0], P[idx,1], P[idx,2],
               c=[NIV_COL[n] for n in niveaux], s=45, depthshade=False)
    ax.scatter(*P[0], c="green", s=55)
    r = np.ptp(P,0); span = max(r.max(),1)*.55; ctr = (P.max(0)+P.min(0))/2
    ax.set_xlim(ctr[0]-span,ctr[0]+span); ax.set_ylim(ctr[1]-span,ctr[1]+span)
    ax.set_zlim(ctr[2]-span,ctr[2]+span); ax.set_box_aspect((1,1,1))
    ax.set_xlabel("N [m]"); ax.set_ylabel("E [m]"); ax.set_zlabel("Up [m]")
    ax.set_title("choregraphie 3D — un point par beat\n(bleu=doux, orange=moyen, rouge=fort)",
                 fontsize=10)

    # force des beats
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(beats, forces, width=.12, color=[NIV_COL[n] for n in niveaux])
    ax.axhline(.33, ls="--", color="grey", lw=.8); ax.axhline(.66, ls="--", color="grey", lw=.8)
    ax.set_title("force de chaque beat", fontsize=10); ax.set_xlabel("t [s]")
    ax.set_ylim(0,1.05); ax.grid(alpha=.3)

    # poussee
    ax = fig.add_subplot(gs[0, 3])
    if res is not None:
        th = np.asarray(res.thrust, float)
        ax.plot(tt, th, color="magenta", lw=1.4)
        ax.axhline(MASS*G, ls=":", color="grey"); ax.axhline(THRUST_MAX, ls="--", color="red")
        ax.set_ylim(0, THRUST_MAX*1.1)
        ok = np.asarray(res.residual_norm).max() <= RESIDUAL_TOL
        ax.set_title(f"poussee — {'VOLABLE' if ok else 'infaisable'}", fontsize=10)
    else:
        ax.text(.5,.5,"f3p_attitude absent",ha="center",va="center",color="#555")
        ax.set_title("poussee", fontsize=10)
    ax.set_xlabel("t [s]"); ax.grid(alpha=.3)

    # angles
    ax = fig.add_subplot(gs[1, 0:2])
    for k,lab,c in [("gamma","pente","#1F77B4"),("chi","cap","#17BECF"),("mu","roulis","#7A5AF8")]:
        ax.plot(tt, np.degrees(seq[k]), color=c, lw=1.4, label=lab)
    for b in beats: ax.axvline(b, color="red", lw=.4, alpha=.45)
    ax.set_title("angles [deg] — les traits rouges sont les beats", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=.3); ax.set_xlabel("t [s]")

    # vitesse
    ax = fig.add_subplot(gs[1, 2:4])
    ax.plot(tt, seq["speed"], color="#00A79F", lw=1.6)
    ax.axhline(3, ls="--", color="red", lw=.8); ax.axhline(7, ls="--", color="red", lw=.8)
    for b in beats: ax.axvline(b, color="red", lw=.4, alpha=.45)
    ax.set_ylim(2.5,7.5); ax.set_title("vitesse [m/s]", fontsize=10)
    ax.grid(alpha=.3); ax.set_xlabel("t [s]")

    # melange + beats
    ax = fig.add_subplot(gs[2, :])
    used = [i for i in range(N_PRIMS) if W[:,i].max() > .02]
    tw = np.linspace(0, tt[-1], len(W))
    ax.stackplot(tw, *[W[:,i] for i in used], labels=[PRIMS[i] for i in used],
                 colors=[COL[PRIMS[i]] for i in used], alpha=.92)
    for b, n in zip(beats, niveaux):
        ax.axvline(b, color="black", lw=1.1 if n=="fort" else .6,
                   alpha=.85 if n=="fort" else .45)
    ax.set_ylim(0,1); ax.set_xlim(0, tw[-1])
    ax.set_title("melange des primitives — chaque trait vertical est un beat "
                 "(epais = beat fort)", fontsize=11, pad=10)
    ax.set_xlabel("t [s]"); ax.set_ylabel("part de chaque primitive")
    ax.legend(loc="lower center", bbox_to_anchor=(.5,-.28), ncol=len(used),
              fontsize=8, frameon=False)

    fig.suptitle(f"Choregraphie F3P pilotee par la musique — {len(seq['choix'])} beats "
                 f"({src})", fontsize=13)
    plt.savefig("music_f3p.png", dpi=100, bbox_inches="tight")
    print(f"  -> music_f3p.png")
    print(f"  {len(seq['choix'])} manoeuvres | encombrement {np.round(P.max(0)-P.min(0),1)} m")
    if res is not None:
        th = np.asarray(res.thrust,float); rn = np.asarray(res.residual_norm,float)
        print(f"  poussee max {th.max():.2f}/{THRUST_MAX:.2f} N ({100*th.max()/THRUST_MAX:.0f}%) "
              f"| residu {rn.max():.1e} | {'VOLABLE' if rn.max()<=RESIDUAL_TOL else 'INFAISABLE'}")


if __name__ == "__main__":
    n = [a for a in sys.argv[1:] if a.isdigit()]
    main(int(n[0]) if n else None)
