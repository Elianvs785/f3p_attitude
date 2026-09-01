"""Validation du montage AVANT d'entrainer.

    python test_setup.py

Verifie trois choses :
  1. la vitesse imposee varie bien selon les intervalles de beats
  2. l'ORACLE (poursuite pure) suit correctement -> l'environnement PERMET le suivi
  3. si f3p_attitude est disponible : la trajectoire de l'oracle est-elle VOLABLE ?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from music_path import path_from_beats, roll_profile_from_music, speed_reference
from track_f3p_env import TrackF3P, N_PRIMS, PRIMS
from oracle import pursuit_action

# cas de test du superviseur : instants tres inegaux
BEATS = np.array([0., 1., 2., 10., 11., 12., 13.5, 15.])
FORCE = np.array([.3, .9, .4, .95, .2, .7, .85, .35])


def build():
    W, T, F = path_from_beats(BEATS, FORCE, shape="rectangle")
    tg = np.linspace(T[0], T[-1], 3000)
    mu = roll_profile_from_music(tg, BEATS, FORCE)
    v, v_seg = speed_reference(W, T, tg)
    env = TrackF3P(dt_ctrl=0.1)
    env.set_path(W, T, lambda t: float(np.interp(t, tg, mu)),
                       lambda t: float(np.interp(t, tg, v)))
    return env, W, T, v_seg, tg, mu


if __name__ == "__main__":
    env, W, T, v_seg, tg, mu = build()
    L = np.linalg.norm(np.diff(W, axis=0), axis=1); dt = np.diff(T)

    print("="*62)
    print("1. LA VITESSE DECOULE DE LA GEOMETRIE ET DES BEATS")
    print("="*62)
    print(f"{'segment':>8}{'longueur':>10}{'duree':>8}{'v = L/dt':>12}")
    print("-"*40)
    for i, (l, d) in enumerate(zip(L, dt)):
        print(f"{i:>8}{l:>10.2f} m{d:>7.1f} s{l/d:>10.2f} m/s")
    print(f"\n  rapport v_max/v_min : {v_seg.max()/v_seg.min():.1f}x")
    print("  -> l'avion doit voler tres lentement sur les longs intervalles.")
    print("     Il le fait en se cabrant (regime harrier) : la poussee le")
    print("     soutient, les ailes ne portent plus. Prevu par le modele.")

    print("\n" + "="*62)
    print("2. L'ENVIRONNEMENT PERMET-IL LE SUIVI ? (oracle)")
    print("="*62)
    rng = np.random.default_rng(0)
    def run(pol):
        env.reset(); done = False
        while not done: _, _, done, _ = env.step(pol())
        return env.report(), env.result()
    r_rand, _ = run(lambda: rng.normal(0, 1.5, env.action_dim))
    r_orc, res = run(lambda: pursuit_action(env, k_ang=4.0, k_mu=4.0))
    sd = np.abs(res["speed_dot"])
    print(f"  acceleration longitudinale max : {sd.max():.1f} m/s2 "
          f"-> poussee ~{0.1*(sd.max()+9.81):.2f} N (bornee par construction)")
    print(f"  {'aleatoire':<22} chemin {r_rand['lat_mean']:5.2f} m | "
          f"waypoints {r_rand['wp_err_mean']:5.2f} m | roulis {r_rand['mu_err_deg']:3.0f} deg")
    print(f"  {'oracle (poursuite)':<22} chemin {r_orc['lat_mean']:5.2f} m | "
          f"waypoints {r_orc['wp_err_mean']:5.2f} m | roulis {r_orc['mu_err_deg']:3.0f} deg")
    print("\n  -> l'oracle suit correctement : on peut entrainer.")
    print("     Le RL devra se situer entre les deux, et peut faire MIEUX que")
    print("     l'oracle : la poursuite pure est myope (elle ne voit qu'un")
    print("     waypoint devant, sans anticiper le virage suivant).")

    print("\n" + "="*62)
    print("3. LA TRAJECTOIRE EST-ELLE VOLABLE ? (solveur du superviseur)")
    print("="*62)
    try:
        from f3p_attitude.solver import solve_trajectory
        from f3p_attitude.constants import THRUST_MAX, RESIDUAL_TOL
        r = solve_trajectory(res["t"], res["gamma"], res["chi"], res["speed"],
                             res["mu"], gamma_dot=res["gamma_dot"],
                             chi_dot=res["chi_dot"], speed_dot=res["speed_dot"])
        th = np.asarray(r.thrust, float)
        al = np.degrees(np.abs(np.asarray(r.alpha, float)))
        rn = np.asarray(r.residual_norm, float)
        ok = rn.max() <= RESIDUAL_TOL
        print(f"  poussee        : {th.min():.2f} - {th.max():.2f} N "
              f"(max {THRUST_MAX:.2f}, soit {100*th.max()/THRUST_MAX:.0f}%)")
        print(f"  angle d'attaque: {al.min():.0f} - {al.max():.0f} deg")
        print(f"     (un fort alpha a basse vitesse est NORMAL : regime harrier,")
        print(f"      pas un decrochage. Le modele autorise jusqu'a 89.9 deg.)")
        print(f"  residu max     : {rn.max():.2e}   (tolerance {RESIDUAL_TOL:.0e})")
        print(f"\n  VERDICT : {'VOLABLE' if ok else 'INFAISABLE'}")
        if not ok:
            bad = np.where(rn > RESIDUAL_TOL)[0]
            print(f"    {len(bad)}/{len(rn)} instants problematiques, "
                  f"vers t = {res['t'][bad[0]]:.1f}s")
    except ImportError:
        print("  f3p_attitude introuvable : lance ce script la ou le paquet du")
        print("  superviseur est importable pour avoir la validation physique.")