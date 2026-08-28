"""Bibliotheque de TRAJECTOIRES DE REFERENCE complexes.

Les references tirees au hasard (Dirichlet) donnent des melanges diffus : les
primitives se compensent et la trajectoire reste quasi rectiligne. Ici on
construit des references STRUCTUREES, avec des melanges francs et soutenus, qui
produisent de vraies manoeuvres.

Chaque reference est definie par une suite de points de controle (les poids), et
le melange interpole en douceur entre eux.

NOTE SUR LES BOUCLES
--------------------
Une boucle complete demanderait gamma > 90 deg, or gamma est borne a +/-90 dans
ce parametrage (au-dela, (gamma, chi) perd son sens : l'avion "passe derriere"
la verticale). Les figures verticales sont donc des montees/descentes jusqu'a la
verticale, pas des loopings.
"""
import numpy as np
from blend import PRIMS, N_PRIMS, blend

I = {p: PRIMS.index(p) for p in PRIMS}


def w(**kw):
    v = np.zeros(N_PRIMS)
    for k, x in kw.items():
        v[I[k]] = x
    return v


# ---------------------------------------------------------------- figures
FIGURES = {
    "s_turn": dict(
        duration=8.0,
        info="virage serre a droite puis a gauche (S)",
        K=[w(straight=1),
           w(turn_right=.9, straight=.1),
           w(turn_right=1),
           w(straight=.6, decel=.4),
           w(turn_left=1),
           w(turn_left=.9, straight=.1),
           w(straight=1)]),

    "climbing_spiral": dict(
        duration=9.0,
        info="spirale montante : virage soutenu + montee",
        K=[w(straight=1),
           w(turn_right=.6, climb_up=.4),
           w(turn_right=.55, climb_up=.45),
           w(turn_right=.6, climb_up=.4),
           w(turn_right=.7, climb_down=.3),
           w(turn_right=.6, climb_down=.4),
           w(straight=1)]),

    "figure_eight": dict(
        duration=10.0,
        info="huit : deux boucles horizontales en sens opposes",
        K=[w(straight=1),
           w(turn_right=1),
           w(turn_right=1),
           w(turn_right=.7, straight=.3),
           w(straight=.5, turn_left=.5),
           w(turn_left=1),
           w(turn_left=1),
           w(turn_left=.7, straight=.3),
           w(straight=1)]),

    "wingover": dict(
        duration=8.0,
        info="renversement : montee, demi-tour en haut, descente",
        K=[w(accel=.5, straight=.5),
           w(climb_up=1),
           w(climb_up=.6, turn_right=.4),
           w(turn_right=.8, decel=.2),
           w(turn_right=.6, climb_down=.4),
           w(climb_down=1),
           w(straight=.6, accel=.4)]),

    "roll_and_turn": dict(
        duration=8.0,
        info="tonneaux enchaines pendant un virage",
        K=[w(straight=1),
           w(roll_right=.7, turn_right=.3),
           w(roll_right=.6, turn_right=.4),
           w(straight=.4, turn_right=.6),
           w(roll_left=.7, turn_right=.3),
           w(roll_left=.6, straight=.4),
           w(straight=1)]),

    "vertical_climb": dict(
        duration=8.0,
        info="montee a la verticale avec tonneau, puis retour",
        K=[w(accel=.6, straight=.4),
           w(climb_up=1),
           w(climb_up=.5, roll_right=.5),
           w(roll_right=.8, decel=.2),
           w(climb_down=.7, roll_right=.3),
           w(climb_down=1),
           w(straight=.5, accel=.5)]),

    "slalom": dict(
        duration=10.0,
        info="slalom : alternance rapide de virages",
        K=[w(straight=1),
           w(turn_right=1), w(turn_left=1),
           w(turn_right=1), w(turn_left=1),
           w(turn_right=1), w(turn_left=1),
           w(straight=1)]),

    "dive_recovery": dict(
        duration=8.0,
        info="piquer, accelerer, ressource avec virage",
        K=[w(straight=1),
           w(climb_down=1),
           w(climb_down=.6, accel=.4),
           w(accel=.5, turn_left=.5),
           w(climb_up=.7, turn_left=.3),
           w(climb_up=.8, decel=.2),
           w(straight=1)]),
}


def make(name, dt=0.02, state0=None):
    """Construit une reference nommee."""
    f = FIGURES[name]
    st = state0 or dict(gamma=0.0, chi=0.0, mu=0.0, speed=5.0)
    ref = blend(np.array(f["K"]), f["duration"], st, dt)
    ref["name"] = name; ref["info"] = f["info"]
    ref["K"] = np.array(f["K"])
    return ref


def make_random(rng, dt=0.02, duration=8.0, n_ctrl=7, sharpness=0.35):
    """Reference aleatoire mais STRUCTUREE : melanges francs (peu de primitives
    dominantes a la fois), donc de vraies manoeuvres.

    sharpness petit -> melanges plus tranches (Dirichlet concentre sur peu de
    primitives). Avec alpha=1 on obtiendrait des melanges diffus qui se
    compensent et donnent une trajectoire presque droite.
    """
    K = rng.dirichlet(np.ones(N_PRIMS)*sharpness, size=n_ctrl)
    st = dict(gamma=0.0, chi=0.0, mu=0.0, speed=5.0)
    ref = blend(K, duration, st, dt)
    ref["name"] = "aleatoire"; ref["info"] = "melange structure aleatoire"
    ref["K"] = K
    return ref