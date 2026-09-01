"""FIGURE du cube v2 — diagnostic ET planche pour le poster.

    python plot_cube.py                 # lit cube_v2_result.npz

Six panneaux, memes couleurs de primitives que plot_music_f3p.py :
  1. trajectoire 3D : RL, oracle, waypoints
  2. ERREUR PAR BEAT (le panneau de diagnostic : quel beat plombe la moyenne ?)
  3. vitesse au cours du temps — le RL ralentit-il AVANT les coins ?
  4. retard/avance sur l'horaire (le potentiel d'horaire fait-il son travail ?)
  5. melange des primitives — la demande du superviseur, lisible contre les beats
  6. courbes d'apprentissage

Le panneau 2 est celui qui compte aujourd'hui : un beat_err moyen de 2.12 m avec
un maximum de 5.45 m signifie qu'UN seul beat plombe tout, et aucune moyenne ne
dira lequel.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from track_music_env import PRIMS

COL = {"straight": "#4A4A4A", "turn_right": "#00A79F", "turn_left": "#7FD8D2",
       "pitch_up": "#E8833A", "pitch_down": "#F5C6A0",
       "roll_right": "#7A5AF8", "roll_left": "#B9A8FB"}
C_RL, C_ORC, C_WP = "#C0392B", "#1F77B4", "#111111"


def beat_errors(P, t, W, T):
    """Ecart au waypoint a l'instant EXACT du beat (le critere du contrat)."""
    out = []
    for i in range(1, len(T)):
        k = int(np.argmin(np.abs(t - T[i])))
        out.append(float(np.linalg.norm(P[k] - W[i])))
    return np.array(out)


def main(path="cube_v2_result.npz"):
    d = np.load(path, allow_pickle=True)
    P_rl, P_or = d["pos_rl"], d["pos_orc"]
    t, W, T = d["t"], d["W"], d["T"]
    spd, Wt, hist = d["speed_rl"], d["weights"], d["hist"]
    e_rl = beat_errors(P_rl, t, W, T)
    e_or = beat_errors(P_or, t[:len(P_or)], W, T)

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9])
    fig.suptitle("Cube v2 — suivi de waypoints musicaux par melange continu",
                 fontsize=14)

    # --- 1. 3D -------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax.plot(*P_or.T, color=C_ORC, lw=1.4, label="oracle")
    ax.plot(*P_rl.T, color=C_RL, lw=1.6, label="RL")
    ax.plot(*W.T, "o--", color=C_WP, ms=6, lw=1.0, alpha=.7, label="waypoints")
    for i, w in enumerate(W):
        ax.text(*w, f" {i}", fontsize=8)
    # axes A L'ECHELLE, sinon un cube parait ecrase et le suivi parait faux
    allp = np.vstack([P_rl, P_or, W])
    ctr = (allp.max(0)+allp.min(0))/2.0
    rad = float((allp.max(0)-allp.min(0)).max())/2.0 * 1.05
    ax.set_xlim(ctr[0]-rad, ctr[0]+rad); ax.set_ylim(ctr[1]-rad, ctr[1]+rad)
    ax.set_zlim(ctr[2]-rad, ctr[2]+rad)
    try: ax.set_box_aspect((1, 1, 1))
    except Exception: pass
    ax.set_xlabel("N [m]"); ax.set_ylabel("E [m]"); ax.set_zlabel("Up [m]")
    ax.set_title("trajectoire 3D (axes a l'echelle)"); ax.legend(fontsize=8)

    # --- 2. erreur PAR BEAT (le diagnostic) --------------------------
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(1, len(T))
    ax.bar(x-0.2, e_or, 0.4, color=C_ORC, label="oracle")
    ax.bar(x+0.2, e_rl, 0.4, color=C_RL, label="RL")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="critere 1 m")
    for i, (a, b) in enumerate(zip(e_or, e_rl), start=1):
        if b > 1.0:
            ax.annotate(f"{b:.1f}", (i+0.2, b), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("numero du beat"); ax.set_ylabel("ecart au waypoint [m]")
    ax.set_title("ERREUR A L'INSTANT DU BEAT\n(quel beat plombe la moyenne ?)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    # --- 3. vitesse --------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t[:len(spd)], spd, color=C_RL, lw=1.4)
    for Ti in T:
        ax.axvline(Ti, color="k", alpha=.25, lw=.8)
    seg_v = np.linalg.norm(np.diff(W, axis=0), axis=1)/np.diff(T)
    for i in range(len(seg_v)):
        ax.hlines(seg_v[i], T[i], T[i+1], color="gray", ls=":", lw=1.2)
    ax.set_xlabel("t [s]"); ax.set_ylabel("vitesse [m/s]")
    ax.set_title("vitesse choisie par le RL\n(pointille gris = moyenne imposee L/dt)")
    ax.grid(alpha=.3)

    # --- 4. retard sur l'horaire -------------------------------------
    ax = fig.add_subplot(gs[0, 3])
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(W, axis=0), axis=1))])
    for P, c, lab in [(P_or, C_ORC, "oracle"), (P_rl, C_RL, "RL")]:
        n = min(len(P), len(t))
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P[:n], axis=0), axis=1))])
        ax.plot(t[:n], s - np.interp(t[:n], T, cum), color=c, lw=1.4, label=lab)
    ax.axhline(0, color="k", lw=1)
    for Ti in T:
        ax.axvline(Ti, color="k", alpha=.25, lw=.8)
    ax.set_xlabel("t [s]"); ax.set_ylabel("avance (+) / retard (-) [m]")
    ax.set_title("ecart a l'horaire des beats"); ax.legend(fontsize=8); ax.grid(alpha=.3)

    # --- 5. melange des primitives (la demande du superviseur) -------
    ax = fig.add_subplot(gs[1, :])
    tw = np.linspace(T[0], T[-1], len(Wt))
    ax.stackplot(tw, *[Wt[:, i] for i in range(len(PRIMS))],
                 colors=[COL[p] for p in PRIMS], labels=PRIMS)
    for Ti in T:
        ax.axvline(Ti, color="k", lw=1.2)
    ax.set_xlim(tw[0], tw[-1]); ax.set_ylim(0, 1)
    ax.set_xlabel("t [s]"); ax.set_ylabel("part de chaque primitive")
    ax.set_title("melange des primitives a chaque instant "
                 "— traits verticaux = beats  (ex : 40 % droit + 60 % virage)")
    ax.legend(ncol=7, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.16))

    fig.tight_layout(rect=[0, .02, 1, .96])
    fig.savefig("cube_v2.png", dpi=130)
    print("  -> cube_v2.png")

    # --- courbes d'apprentissage, planche separee --------------------
    if hist.size:
        f2, axs = plt.subplots(1, 3, figsize=(13, 3.4))
        for ax_, j, lab in [(axs[0], 0, "score / pas"),
                            (axs[1], 1, "beat_err [m]"),
                            (axs[2], 2, "ecart au chemin [m]")]:
            if j < hist.shape[1]:
                ax_.plot(hist[:, j], color=C_RL)
                ax_.set_xlabel("iteration"); ax_.set_title(lab); ax_.grid(alpha=.3)
        axs[1].axhline(1.0, color="k", ls="--", lw=1)
        f2.tight_layout(); f2.savefig("cube_v2_learning.png", dpi=130)
        print("  -> cube_v2_learning.png")

    # --- resume chiffre ----------------------------------------------
    print()
    print(f"{'beat':>6}{'oracle':>10}{'RL':>10}")
    print("-"*26)
    for i, (a, b) in enumerate(zip(e_or, e_rl), start=1):
        flag = "  <-- pire" if b == e_rl.max() else ""
        print(f"{i:>6}{a:>9.2f}m{b:>9.2f}m{flag}")
    print(f"{'moyenne':>6}{e_or.mean():>9.2f}m{e_rl.mean():>9.2f}m")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cube_v2_result.npz")
