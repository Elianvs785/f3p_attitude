"""Visualisation du SUIVI DE TRAJECTOIRE par melange.

Six panneaux :
  1. 3D : reference (pointilles) vs trajectoire suivie (trait plein)
  2-4. gamma(t), chi(t), speed(t) : reference vs suivi
  5. erreur de suivi dans le temps
  6. LE panneau cle : repartition des poids a chaque instant
     -> on lit directement "a t=2.3s, c'est 62 % turn_right + 25 % climb_up + 13 % straight"

    python plot_track.py           # utilise ppo_track.pt s'il existe
    python plot_track.py --oracle  # trace le suivi ORACLE (borne basse)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from track_env import TrackEnv, reference_from_weights
from blend import PRIMS, N_PRIMS, weights_over_time

COL = {"straight":"#4A4A4A","turn_right":"#00A79F","turn_left":"#7FD8D2",
       "climb_up":"#E8833A","climb_down":"#F5C6A0","roll_right":"#7A5AF8",
       "roll_left":"#B9A8FB","accel":"#2E8B57","decel":"#A03030"}


def positions(t, gamma, chi, speed):
    v = np.stack([speed*np.cos(gamma)*np.cos(chi),
                  speed*np.cos(gamma)*np.sin(chi),
                  speed*np.sin(gamma)], 1)
    p = np.zeros_like(v)
    for i in range(len(t)-1):
        p[i+1] = p[i] + 0.5*(v[i]+v[i+1])*(t[i+1]-t[i])
    return p


def run_policy(env, ref, policy):
    """policy(obs) -> action. Retourne la trajectoire suivie."""
    env.set_reference(ref)
    s = env.reset(); done = False
    while not done:
        s, r, done, _ = env.step(policy(s, env))
    return env.result()


def plot(env, ref, out, titre, path, mark_fracs=(0.15, 0.4, 0.65, 0.9)):
    t_r = ref["t"]
    t_o = out["t"]
    P_r = positions(t_r, ref["gamma"], ref["chi"], ref["speed"])
    P_o = positions(t_o, out["gamma"], out["chi"], out["speed"])
    W = out["weights"]
    t_w = np.linspace(0, t_o[-1], len(W))

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 0.85, 1.25], hspace=0.55, wspace=0.30)

    # --- 3D ---
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax.plot(P_r[:,0], P_r[:,1], P_r[:,2], "--", color="black", lw=1.8, label="reference")
    ax.plot(P_o[:,0], P_o[:,1], P_o[:,2], color="#00A79F", lw=2.2, label="suivie")
    ax.scatter(*P_r[0], c="green", s=45)
    allP = np.vstack([P_r, P_o]); r = np.ptp(allP, 0)
    span = max(r.max(), 1.0)*0.6; ctr = (allP.max(0)+allP.min(0))/2
    ax.set_xlim(ctr[0]-span, ctr[0]+span); ax.set_ylim(ctr[1]-span, ctr[1]+span)
    ax.set_zlim(ctr[2]-span, ctr[2]+span); ax.set_box_aspect((1,1,1))
    ax.set_xlabel("N [m]"); ax.set_ylabel("E [m]"); ax.set_zlabel("Up [m]")
    ax.set_title("trajectoire 3D", fontsize=10); ax.legend(fontsize=7)

    # --- gamma / chi / speed ---
    for k, (key, lab, unit) in enumerate([("gamma","pente gamma","deg"),
                                          ("chi","cap chi","deg"),
                                          ("speed","vitesse","m/s")]):
        ax = fig.add_subplot(gs[0, k+1])
        f = np.degrees if unit == "deg" else (lambda x: x)
        ax.plot(t_r, f(ref[key]), "--", color="black", lw=1.6, label="reference")
        ax.plot(t_o, f(out[key]), color="#00A79F", lw=1.8, label="suivie")
        ax.set_title(f"{lab} [{unit}]", fontsize=10); ax.grid(alpha=.3)
        ax.set_xlabel("t [s]")
        if k == 0: ax.legend(fontsize=7)

    # --- erreur de suivi ---
    ax = fig.add_subplot(gs[1, 0:2])
    eg, ec, es = [], [], []
    for i, ti in enumerate(t_o):
        j = int(np.clip(np.searchsorted(t_r, ti), 0, len(t_r)-1))
        eg.append(np.degrees(abs(out["gamma"][i]-ref["gamma"][j])))
        d = out["chi"][i]-ref["chi"][j]
        ec.append(np.degrees(abs(np.arctan2(np.sin(d), np.cos(d)))))
        es.append(abs(out["speed"][i]-ref["speed"][j]))
    ax.plot(t_o, eg, label="pente [deg]", lw=1.4)
    ax.plot(t_o, ec, label="cap [deg]", lw=1.4)
    ax.plot(t_o, es, label="vitesse [m/s]", lw=1.4)
    ax.set_title(f"erreur de suivi\n(moy : {np.mean(eg):.1f} deg / {np.mean(ec):.1f} deg / {np.mean(es):.2f} m/s)",
                 fontsize=9)
    ax.grid(alpha=.3); ax.legend(fontsize=7); ax.set_xlabel("t [s]")

    # --- mu (roulis) ---
    axm = fig.add_subplot(gs[1, 2:4])
    axm.plot(t_o, np.degrees(out["mu"]), color="#7A5AF8", lw=1.6, label="mu suivi")
    if "mu" in ref:
        axm.plot(t_r, np.degrees(ref["mu"]), "--", color="black", lw=1.4, label="mu reference")
    axm.set_title("roulis mu [deg]", fontsize=10); axm.grid(alpha=.3)
    axm.set_xlabel("t [s]"); axm.legend(fontsize=7)

    # --- LE panneau cle : repartition des poids ---
    ax = fig.add_subplot(gs[2, :])
    used = [i for i in range(N_PRIMS) if W[:, i].max() > 0.02]
    ax.stackplot(t_w, *[W[:, i] for i in used], labels=[PRIMS[i] for i in used],
                 colors=[COL[PRIMS[i]] for i in used], alpha=.92)
    ax.set_ylim(0, 1); ax.set_xlim(0, t_w[-1])
    ax.set_xlabel("t [s]"); ax.set_ylabel("part de chaque primitive")
    ax.set_title("Repartition choisie a chaque instant pour suivre la reference",
                 fontsize=11, pad=32)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=len(used),
              fontsize=8, frameon=False)
    for fr in mark_fracs:
        ts = fr*t_w[-1]
        i = int(np.clip(np.searchsorted(t_w, ts), 0, len(W)-1))
        w = W[i]; order = np.argsort(-w)[:3]
        lbl = " + ".join(f"{100*w[j]:.0f}% {PRIMS[j]}" for j in order if w[j] > 0.05)
        ax.axvline(ts, color="black", lw=1.0, alpha=.75)
        ax.text(ts, 1.015, f"t={ts:.1f}s\n{lbl}", ha="center", va="bottom",
                fontsize=8, family="monospace")

    fig.suptitle(titre, fontsize=13)
    plt.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {path}")
    print(f"     erreur moyenne : pente {np.mean(eg):.1f} deg, cap {np.mean(ec):.1f} deg, "
          f"vitesse {np.mean(es):.2f} m/s")
    return path


if __name__ == "__main__":
    from references import FIGURES, make, make_random
    rng = np.random.default_rng(3)
    env = TrackEnv(dt_ctrl=0.25)
    # choix de la figure : python plot_track.py wingover
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    fig_name = names[0] if names else "climbing_spiral"
    if fig_name == "random":
        ref = make_random(rng, duration=8.0)
    else:
        if fig_name not in FIGURES:
            print("figures disponibles :", ", ".join(FIGURES)); sys.exit(1)
        ref = make(fig_name)
    K = ref["K"]
    print(f"reference : {ref['name']} — {ref['info']}")

    use_oracle = "--oracle" in sys.argv or not os.path.exists("ppo_track.pt")
    if use_oracle:
        print("politique ORACLE (rejoue les poids de la reference)")
        def pol(s, e):
            tnow = e.k*e.dt_ctrl + e.dt_ctrl*0.5
            return np.log(np.maximum(weights_over_time(K, np.array([tnow]), float(ref["t"][-1]))[0], 1e-9))
        titre = f"Suivi ORACLE — {ref['name']} : {ref['info']}"
        out_name = f"track_{ref['name']}_oracle.png"
    else:
        import torch
        from ppo_track import ActorCritic
        net = ActorCritic(env.state_dim, env.action_dim)
        net.load_state_dict(torch.load("ppo_track.pt"))
        net.eval()
        print("politique PPO apprise (ppo_track.pt)")
        def pol(s, e):
            with torch.no_grad():
                return net.pi(torch.as_tensor(s, dtype=torch.float32)).numpy()
        titre = f"Suivi appris par PPO — {ref['name']} : {ref['info']}"
        out_name = f"track_{ref['name']}_ppo.png"

    out = run_policy(env, ref, pol)
    plot(env, ref, out, titre, out_name)