"""COMPOSITION de primitives en figures F3P.

L'IDEE DU SUPERVISEUR : "au milieu du virage, 40% droit 60% virage".
Un segment de duree D est reparti entre plusieurs primitives selon des POIDS.
    mix(D, [("straight", 0.4), ("turn", 0.6)], params)
donne 0.4*D de ligne droite puis 0.6*D de virage, enchaines sans discontinuite.

C'est cette repartition que le RL apprendra a choisir.
"""
import numpy as np
from primitives import PRIMITIVES


def initial_state(speed=5.0, gamma=0.0, chi=0.0, mu=0.0):
    return dict(gamma=gamma, chi=chi, speed=speed, mu=mu)


def _end_state(seg):
    return dict(gamma=float(seg["gamma"][-1]), chi=float(seg["chi"][-1]),
                speed=float(seg["speed"][-1]), mu=float(seg["mu"][-1]))


def sequence(state, steps, dt=0.02):
    """Enchaine des primitives. steps = [(nom, duree, kwargs), ...].

    Chaque primitive PART de l'etat laisse par la precedente : c'est ce qui
    rend la sequence continue.
    """
    t_all = []; parts = {k: [] for k in ("gamma","gamma_dot","chi","chi_dot",
                                         "speed","speed_dot","mu","mu_dot")}
    t0 = 0.0; st = dict(state); marks = []
    for name, dur, kw in steps:
        seg = PRIMITIVES[name](st, dur, dt=dt, **kw)
        keep = slice(1, None) if t_all else slice(None)   # eviter de doubler le point de jonction
        t_all.append(seg["t"][keep] + t0)
        for k in parts: parts[k].append(seg[k][keep])
        marks.append((t0, t0 + float(seg["t"][-1]), name))
        t0 += float(seg["t"][-1]); st = _end_state(seg)
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["t"] = np.concatenate(t_all); out["marks"] = marks
    return out


def mix(state, duration, weights, params=None, dt=0.02):
    """Repartit `duration` entre plusieurs primitives selon des POIDS.

        mix(st, 2.0, [("straight", 0.4), ("turn", 0.6)], {"turn": {"delta_chi_deg": 90}})
        -> 0.8 s de ligne droite puis 1.2 s de virage a 90 deg

    C'est le "40% droit, 60% virage" du superviseur.
    """
    params = params or {}
    tot = sum(w for _, w in weights)
    steps = [(name, duration*w/tot, dict(params.get(name, {}))) for name, w in weights]
    return sequence(state, steps, dt=dt)


def check_continuity(seq, tol=1e-9):
    """Verifie qu'il n'y a aucun SAUT AUX JONCTIONS entre primitives.

    Attention : comparer des points consecutifs sur toute la trajectoire ne dit
    rien (pendant un virage chi varie normalement a chaque pas). On compare donc
    l'ecart A LA JONCTION a l'ecart typique juste avant/apres.

    Par construction :
      - les VALEURS sont continues (chaque primitive part de l'etat final de la
        precedente)
      - les DERIVEES aussi, car smoothstep a une derivee nulle en 0 et en 1
    """
    t = seq["t"]
    # indices des jonctions
    junc = []
    acc = 0.0
    for a, b, _ in seq["marks"][:-1]:
        acc = b
        junc.append(int(np.searchsorted(t, acc)))
    res = {"n_jonctions": len(junc)}
    worst_val = worst_dot = 0.0
    for k in ("gamma", "chi", "speed", "mu"):
        y = seq[k]; yd = seq[k + "_dot"]
        for j in junc:
            if 2 <= j < len(y) - 2:
                # saut a la jonction compare a la variation locale typique
                loc = max(abs(y[j-1]-y[j-2]), abs(y[j+2]-y[j+1]), 1e-12)
                worst_val = max(worst_val, abs(y[j]-y[j-1]) / loc)
                # la derivee doit passer par ~0 a la jonction (smoothstep)
                worst_dot = max(worst_dot, min(abs(yd[j-1]), abs(yd[j+1])))
    res["saut_relatif_max"] = float(worst_val)   # ~1 = pas de saut anormal
    res["derivee_aux_jonctions"] = float(worst_dot)  # ~0 attendu
    res["ok"] = bool(worst_val < 3.0 and worst_dot < 1e-6)
    return res


def to_solver_input(seq):
    """Format attendu par solve_trajectory() du superviseur."""
    return dict(t=seq["t"], gamma=seq["gamma"], chi=seq["chi"],
                speed=seq["speed"], mu=seq["mu"],
                gamma_dot=seq["gamma_dot"], chi_dot=seq["chi_dot"],
                speed_dot=seq["speed_dot"])
