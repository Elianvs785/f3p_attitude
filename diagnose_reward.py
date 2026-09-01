"""DIAGNOSTIC : de quoi la recompense est-elle faite ?

Le RL a appris le roulis (6 deg, comme l'oracle) et rien sur la position
(12.6 m contre 0.52 m). Hypothese : le terme de position est INVISIBLE — a
grande distance, exp(-1.5*e_lat) est numeriquement nul, donc s'approcher ne
rapporte rien de mesurable et il n'y a pas de gradient a remonter.

On ne corrige pas avant d'avoir verifie. On decompose la recompense en ses
quatre termes, pour l'aleatoire (ce que voit le RL au depart) et pour l'oracle
(ce qu'il devrait viser).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from test_setup import build
from oracle import pursuit_action


def decompose(env, pol):
    """Rejoue un episode en recalculant chaque terme separement."""
    env.reset(); done = False
    acc = dict(pos=[], time=[], mu=[], speed=[], lat=[])
    while not done:
        i_before = env._target_index(env.t)
        t_before = env.t
        _, _, done, _ = env.step(pol())
        i = env._target_index(env.t)
        proj, _ = env._closest_on_path(env.pos, i)
        e_lat = float(np.linalg.norm(env.pos - proj))
        acc["lat"].append(e_lat)
        acc["pos"].append(env.w_pos*np.exp(-1.5*e_lat))
        rt = 0.0
        if t_before < env.T[i_before] <= env.t:
            n_sub = int(round(env.dt_ctrl/env.dt)) + 1
            P = np.asarray(env.log["pos"][-n_sub:])
            rt = env.w_time*np.exp(-1.0*float(
                np.linalg.norm(P - env.W[i_before], axis=1).min()))
        acc["time"].append(rt)
        dmu = env.mu - float(env.mu_ref_fn(env.t))
        acc["mu"].append(env.w_mu*np.exp(
            -1.5*abs(np.arctan2(np.sin(dmu), np.cos(dmu)))))
        v_ref = float(env.v_ref_fn(env.t))
        acc["speed"].append(0.3*np.exp(-2.0*abs(env.speed - v_ref)/max(v_ref, 0.5)))
    return {k: np.asarray(v) for k, v in acc.items()}


if __name__ == "__main__":
    env, W, T, v_seg, tg, mu = build()
    rng = np.random.default_rng(0)
    A = decompose(env, lambda: rng.normal(0, 1.5, env.action_dim))
    O = decompose(env, lambda: pursuit_action(env, k_ang=4.0, k_mu=4.0))

    print("="*66)
    print("1. CONTRIBUTION MOYENNE DE CHAQUE TERME (par pas)")
    print("="*66)
    print(f"{'terme':<12}{'aleatoire':>14}{'oracle':>14}{'ecart':>14}")
    print("-"*54)
    for k in ("pos", "time", "mu", "speed"):
        a, o = A[k].mean(), O[k].mean()
        print(f"{k:<12}{a:>14.2e}{o:>14.2e}{o-a:>+14.2e}")
    ta = sum(A[k].mean() for k in ("pos", "time", "mu", "speed"))
    to = sum(O[k].mean() for k in ("pos", "time", "mu", "speed"))
    print(f"{'TOTAL':<12}{ta:>14.3f}{to:>14.3f}{to-ta:>+14.3f}")

    print("\n" + "="*66)
    print("2. LE TERME DE POSITION A-T-IL UN GRADIENT ?")
    print("="*66)
    print(f"  ecart au chemin, aleatoire : median {np.median(A['lat']):.1f} m, "
          f"max {A['lat'].max():.1f} m")
    print(f"  valeur de exp(-1.5*e_lat) a ces distances :\n")
    print(f"{'e_lat (m)':>12}{'recompense':>16}{'gain si -1 m':>16}")
    print("-"*44)
    for d in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 15.0):
        r = np.exp(-1.5*d)
        gain = np.exp(-1.5*max(d-1, 0)) - r
        print(f"{d:>12.1f}{r:>16.2e}{gain:>16.2e}")
    print("\n  -> a partir de ~5 m, se rapprocher d'un metre entier ne change")
    print("     rien de mesurable. Le RL part a 15 m : il est dans le plat.")

    print("\n" + "="*66)
    print("3. PART DE CHAQUE TERME DANS CE QUE LE RL PEUT GAGNER")
    print("="*66)
    tot = sum(max(O[k].mean()-A[k].mean(), 0) for k in ("pos","time","mu","speed"))
    for k in ("pos", "time", "mu", "speed"):
        g = max(O[k].mean()-A[k].mean(), 0)
        print(f"  {k:<8} {100*g/max(tot,1e-12):>5.1f} %")
    print("\n  Si 'mu' domine, l'agent optimise le roulis et ignore le reste :")
    print("  c'est exactement ce qu'on observe (6 deg de roulis, 12.6 m d'ecart).")