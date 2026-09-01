"""VERDICT DU SOLVEUR — la trajectoire volee est-elle VOLABLE ?

Le contrat dit que chaque jalon se termine par un verdict physique. On a une
trajectoire precise (0.09 m au beat), on ne sait toujours pas si un avion peut
la voler.

METHODE. L'env enregistre gamma, chi, speed, mu et leurs derivees a chaque
sous-pas. On les passe a `solve_trajectory`, qui cherche pour chaque instant les
commandes (alpha, beta, poussee) equilibrant les forces.

LE JUGE EST LE RESIDU, pas `r.success` (qui dit seulement que least_squares a
converge vers un minimum local, pas que ce minimum vaut zero) ni la poussee
(bornee par construction dans le solveur, donc jamais depassee).

    python verify_flight.py              # regulier + minsnap
    STRESS=1 python verify_flight.py     # extreme + polyline
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from track_music_env import TrackMusic, oracle_action
from music_path import path_from_beats, roll_profile_from_music
from f3p_attitude.solver import solve_trajectory
from f3p_attitude.constants import THRUST_MAX, RESIDUAL_TOL

STRESS = bool(int(os.environ.get("STRESS", 0)))
SUB = int(os.environ.get("SUB", 3))          # 1 instant sur SUB (cout du solveur)


def build():
    W_, T_ = None, None
    from music_path import cube_3d
    W_ = cube_3d()
    T_ = (np.array([0., 1., 2., 10., 11., 12., 13.5, 15., 16.5])
          if STRESS else np.arange(9)*2.0)
    F = np.array([.3, .9, .4, .95, .2, .7, .85, .35, .6])
    tg = np.linspace(T_[0], T_[-1], 3000)
    mu = roll_profile_from_music(tg, T_, F)
    env = TrackMusic()
    env.set_path(W_, T_, lambda t: float(np.interp(t, tg, mu)),
                 reference=("polyline" if STRESS else "minsnap"))
    return env, W_, T_


def fly(env, policy):
    env.reset(); done = False
    while not done:
        _, _, done, _ = env.step(policy())
    return env.result(), env.report()


def judge(res, label):
    """Passe la trajectoire au solveur et compte les instants infaisables."""
    k = slice(None, None, SUB)
    t = np.asarray(res["t"])[k]
    sp = np.asarray(res["speed"])[k]
    r = solve_trajectory(t, np.asarray(res["gamma"])[k], np.asarray(res["chi"])[k],
                         sp, np.asarray(res["mu"])[k],
                         gamma_dot=np.asarray(res["gamma_dot"])[k],
                         chi_dot=np.asarray(res["chi_dot"])[k],
                         speed_dot=np.gradient(sp, t))
    rn = np.asarray(r.residual_norm, float)
    th = np.asarray(r.thrust, float)
    al = np.degrees(np.asarray(r.alpha, float))
    bad = np.where(rn > RESIDUAL_TOL)[0]
    ok = len(bad) == 0
    print(f"\n--- {label}")
    print(f"  poussee   : {th.min():.2f} - {th.max():.2f} N "
          f"({100*th.max()/THRUST_MAX:.0f} % du max)")
    print(f"  alpha     : {al.min():.0f} - {al.max():.0f} deg")
    print(f"  residu max: {rn.max():.2e}   (tolerance {RESIDUAL_TOL:.0e})")
    print(f"  VERDICT   : {'VOLABLE' if ok else 'INFAISABLE'}"
          f"   {len(bad)}/{len(rn)} instants au-dessus de la tolerance")
    if not ok:
        print(f"  fenetres fautives : t = "
              f"{', '.join(f'{x:.1f}' for x in np.unique(np.round(t[bad], 1))[:12])} s")
    return rn, th, t


if __name__ == "__main__":
    env, W, T = build()
    print(f"cube {'STRESS' if STRESS else 'regulier'} | reference {env.reference}")

    res_o, rep_o = fly(env, lambda: oracle_action(env))
    print(f"\noracle : beat_err {rep_o['beat_err']:.2f} m | suivi {rep_o['track_err']:.2f} m")
    judge(res_o, "ORACLE")

    # le RL entraine, s'il existe et si torch est disponible
    try:
        import torch
        from ppo_core import ActorCritic
        net = ActorCritic(env.state_dim, env.action_dim)
        net.load_state_dict(torch.load("ppo_cube_v2.pt", map_location="cpu"))
        net.eval()

        def pol():
            with torch.no_grad():
                d = net.pi(torch.as_tensor(env._obs(), dtype=torch.float32)).numpy()
            return oracle_action(env) + d

        res_r, rep_r = fly(env, pol)
        print(f"\nRL     : beat_err {rep_r['beat_err']:.2f} m | suivi {rep_r['track_err']:.2f} m")
        judge(res_r, "RL (residuel)")
    except FileNotFoundError:
        print("\n(ppo_cube_v2.pt absent : lancer train_cube.py d'abord)")
    except ImportError:
        print("\n(torch absent : verdict de l'oracle seulement)")

    print("\nRappel : le solveur repond 'ces commandes existent', pas "
          "'un controleur peut les tenir'.\n(bilan de FORCES seulement : ni "
          "moments, ni inertie de roulis)")
