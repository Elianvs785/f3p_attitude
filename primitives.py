"""Bibliotheque de PRIMITIVES de vol F3P.

FORMAT
------
Une figure de vol est decrite, comme dans scenarios.py du superviseur, par
quatre profils temporels :
    gamma  : pente de la trajectoire  (0 = horizontal, +90 deg = montee verticale)
    chi    : cap                      (direction dans le plan horizontal)
    speed  : vitesse
    mu     : roulis autour du vecteur vitesse (0 = normal, 90 deg = vol tranche)
plus leurs derivees, dont le solveur a besoin.

DIFFERENCE AVEC scenarios.py
----------------------------
Ses scenarios sont ABSOLUS (gamma=0, chi=0, speed=5 en dur) : on ne peut pas
les enchainer. Ici les primitives sont RELATIVES : chacune part de l'etat
laisse par la precedente et applique un CHANGEMENT. C'est ce qui permet de les
composer en sequences.

TRANSITIONS
-----------
Tous les changements utilisent smoothstep (repris de scenarios.py) :
    s(u) = 3u^2 - 2u^3,   s(0)=0, s(1)=1, s'(0)=s'(1)=0
Les derivees s'annulent aux extremites, donc deux primitives s'enchainent SANS
discontinuite de derivee. C'est exactement le smoothstep01 du superviseur.
"""
import numpy as np

# etat de vol : (gamma, chi, speed, mu)
State = dict


def smoothstep(u):
    """s(u) = 3u^2 - 2u^3 (identique a smoothstep01 de scenarios.py)."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def dsmoothstep(u):
    """s'(u) = 6u(1-u) — nulle en 0 et 1, d'ou la continuite des derivees."""
    u = np.clip(u, 0.0, 1.0)
    return 6.0 * u * (1.0 - u)


def _ramp(t_local, duration, v0, v1):
    """Transition douce de v0 a v1 sur `duration`. Retourne (valeurs, derivees)."""
    if duration <= 0:
        return np.full_like(t_local, v1), np.zeros_like(t_local)
    u = t_local / duration
    val = v0 + (v1 - v0) * smoothstep(u)
    dot = (v1 - v0) * dsmoothstep(u) / duration
    return val, dot


def _hold(t_local, v):
    return np.full_like(t_local, v), np.zeros_like(t_local)


def _make(t_local, g, gd, c, cd, s, sd, m, md):
    return dict(t=t_local, gamma=g, gamma_dot=gd, chi=c, chi_dot=cd,
                speed=s, speed_dot=sd, mu=m, mu_dot=md)


# ---------------------------------------------------------------- primitives

def straight(state, duration, dt=0.02, speed=None):
    """Vol rectiligne : tout est maintenu. `speed` permet d'accelerer en douceur."""
    t = np.arange(0.0, duration + dt*0.5, dt)
    g, gd = _hold(t, state["gamma"]); c, cd = _hold(t, state["chi"])
    m, md = _hold(t, state["mu"])
    if speed is None: s, sd = _hold(t, state["speed"])
    else:             s, sd = _ramp(t, duration, state["speed"], speed)
    return _make(t, g, gd, c, cd, s, sd, m, md)


def turn(state, duration, delta_chi_deg, dt=0.02, speed=None):
    """VIRAGE : le cap change de delta_chi (positif = vers la droite)."""
    t = np.arange(0.0, duration + dt*0.5, dt)
    g, gd = _hold(t, state["gamma"])
    c, cd = _ramp(t, duration, state["chi"], state["chi"] + np.deg2rad(delta_chi_deg))
    m, md = _hold(t, state["mu"])
    if speed is None: s, sd = _hold(t, state["speed"])
    else:             s, sd = _ramp(t, duration, state["speed"], speed)
    return _make(t, g, gd, c, cd, s, sd, m, md)


def climb(state, duration, delta_gamma_deg, dt=0.02, speed=None):
    """MONTEE / DESCENTE : la pente change (delta=+90 -> vertical, comme son
    scenario_level_to_vertical)."""
    t = np.arange(0.0, duration + dt*0.5, dt)
    g, gd = _ramp(t, duration, state["gamma"], state["gamma"] + np.deg2rad(delta_gamma_deg))
    c, cd = _hold(t, state["chi"]); m, md = _hold(t, state["mu"])
    if speed is None: s, sd = _hold(t, state["speed"])
    else:             s, sd = _ramp(t, duration, state["speed"], speed)
    return _make(t, g, gd, c, cd, s, sd, m, md)


def knife_edge(state, duration, dt=0.02, mu_deg=90.0, blend=0.35):
    """VOL TRANCHE : on roule a mu, on tient, on revient (comme son scenario_knife_edge
    mais en version enchainable)."""
    t = np.arange(0.0, duration + dt*0.5, dt)
    d_in = duration*blend; d_out = duration*blend
    m = np.zeros_like(t); md = np.zeros_like(t)
    mu0 = state["mu"]; mu1 = np.deg2rad(mu_deg)
    for i, ti in enumerate(t):
        if ti < d_in:
            v, dv = _ramp(np.array([ti]), d_in, mu0, mu1); m[i], md[i] = v[0], dv[0]
        elif ti > duration - d_out:
            v, dv = _ramp(np.array([ti-(duration-d_out)]), d_out, mu1, mu0); m[i], md[i] = v[0], dv[0]
        else:
            m[i], md[i] = mu1, 0.0
    g, gd = _hold(t, state["gamma"]); c, cd = _hold(t, state["chi"])
    s, sd = _hold(t, state["speed"])
    return _make(t, g, gd, c, cd, s, sd, m, md)


def roll(state, duration, n_turns=1.0, dt=0.02):
    """TONNEAU : mu tourne de n_turns tours complets."""
    t = np.arange(0.0, duration + dt*0.5, dt)
    m, md = _ramp(t, duration, state["mu"], state["mu"] + 2*np.pi*n_turns)
    g, gd = _hold(t, state["gamma"]); c, cd = _hold(t, state["chi"])
    s, sd = _hold(t, state["speed"])
    return _make(t, g, gd, c, cd, s, sd, m, md)


PRIMITIVES = {"straight": straight, "turn": turn, "climb": climb,
              "knife_edge": knife_edge, "roll": roll}
