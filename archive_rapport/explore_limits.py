"""EXPLORATION DES LIMITES PHYSIQUES du F3P.

But : trouver, pour chaque primitive, jusqu'ou on peut aller avant que le
solveur du superviseur declare la figure involable. Ces plages definiront
l'espace de recherche du RL — inutile de le laisser proposer des figures
physiquement impossibles.

Criteres d'infaisabilite :
  - poussee demandee > THRUST_MAX (le moteur ne suit pas)
  - angle d'attaque > ~25 deg (decrochage)
  - le solveur ne converge pas (equilibre des forces impossible)

    python explore_limits.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from compose import initial_state, sequence, to_solver_input
from f3p_attitude.solver import solve_trajectory
from f3p_attitude.constants import THRUST_MAX, THRUST_MIN, MASS, G, RESIDUAL_TOL

ALPHA_STALL = 25.0     # deg


def evaluate(steps, speed, dt=0.05, roll_rate_max_deg=720.0):
    """Construit la figure, la passe au solveur, renvoie les indicateurs.

    CRITERE CORRECT DE FAISABILITE : le RESIDU de l'equilibre des forces.
    Attention aux deux pieges :
      1. r.success vient de least_squares : il dit que l'OPTIMISEUR a converge,
         pas que la physique est satisfaite.
      2. la poussee est BORNEE dans le solveur (bounds=THRUST_MAX), donc elle ne
         depasse JAMAIS le maximum : tester "thrust <= THRUST_MAX" ne teste rien.
    Quand une figure est infaisable, le solveur sature la poussee et laisse un
    RESIDU important. C'est pour cela que constants.py definit RESIDUAL_TOL.

    Limite du modele : le solveur fait un bilan de FORCES seulement (ni moments,
    ni inertie de roulis). Faire tourner mu ne coute donc rien -> on borne la
    vitesse de roulis a la main.
    """
    st = initial_state(speed=speed)
    seq = sequence(st, steps, dt=dt)
    try:
        r = solve_trajectory(**to_solver_input(seq))
    except Exception as e:
        return dict(ok=False, why=f"solveur: {type(e).__name__}", thrust=np.nan,
                    alpha=np.nan, res=np.nan)
    th = np.asarray(r.thrust, float)
    al = np.degrees(np.abs(np.asarray(r.alpha, float)))
    res = np.asarray(r.residual_norm, float)
    thrust_ratio = float(th.max() / THRUST_MAX)
    roll_rate = float(np.degrees(np.abs(seq["mu_dot"]).max()))

    ok, why = True, ""
    if res.max() > RESIDUAL_TOL:
        ok, why = False, "equilibre impossible"      # <- le vrai critere
    elif al.max() > ALPHA_STALL:
        ok, why = False, "decrochage"
    elif roll_rate > roll_rate_max_deg:
        ok, why = False, f"roulis {roll_rate:.0f} deg/s"
    return dict(ok=ok, why=why, thrust=thrust_ratio, alpha=float(al.max()),
                res=float(res.max()))


def sweep(titre, sous_titre, configs, colname):
    print("\n" + "=" * 68)
    print(titre)
    print(sous_titre)
    print("=" * 68)
    print(f"{colname:>16}{'poussee':>10}{'alpha':>9}{'residu':>11}   verdict")
    print("-" * 68)
    limite = None
    for label, steps, speed in configs:
        r = evaluate(steps, speed)
        verdict = "OK" if r["ok"] else f"NON ({r['why']})"
        print(f"{label:>16}{100*r['thrust']:>9.0f}%{r['alpha']:>8.0f}d"
              f"{r['res']:>11.1e}   {verdict}")
        if r["ok"]:
            limite = label
    if limite is not None:
        print(f"  -> limite atteignable : {limite}")
    return limite


if __name__ == "__main__":
    print(f"critere de faisabilite : residu < {RESIDUAL_TOL:.0e} (equilibre des forces)")
    print(f"F3P : masse {MASS*1000:.0f} g | poussee max {THRUST_MAX:.2f} N "
          f"| poussee/poids {THRUST_MAX/(MASS*G):.1f} | decrochage suppose {ALPHA_STALL:.0f} deg")

    # ---- 1. vol en palier : quelle plage de vitesse ? ----
    sweep("1. VOL EN PALIER — plage de vitesse utilisable",
          "   (trop lent : les ailes ne portent plus / trop rapide : trainee)",
          [(f"{v:.1f} m/s", [("straight", 2.0, {})], v) for v in
           (1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0)],
          "vitesse")

    # ---- 2. virage : jusqu'ou serrer ? ----
    sweep("2. VIRAGE 90 deg — duree minimale",
          "   (plus la duree est courte, plus le virage est serre)",
          [(f"{d:.2f} s", [("turn", d, {"delta_chi_deg": 90.0})], 5.0) for d in
           (2.0, 1.5, 1.0, 0.7, 0.5, 0.35, 0.25, 0.15)],
          "duree virage")

    # ---- 3. virage : influence de la vitesse ----
    sweep("3. VIRAGE 90 deg en 0.5 s — influence de la vitesse",
          "   (a vitesse elevee, un virage serre demande plus de force)",
          [(f"{v:.1f} m/s", [("turn", 0.5, {"delta_chi_deg": 90.0})], v) for v in
           (3.0, 4.0, 5.0, 7.0, 9.0, 12.0)],
          "vitesse")

    # ---- 4. montee verticale ----
    sweep("4. PASSAGE A LA VERTICALE (0 -> 90 deg) — duree minimale",
          "   (le scenario level_to_vertical du superviseur, accelere)",
          [(f"{d:.2f} s", [("straight", 0.5, {}), ("climb", d, {"delta_gamma_deg": 90.0}),
                           ("straight", 0.5, {})], 5.0) for d in
           (2.0, 1.5, 1.0, 0.7, 0.5, 0.35, 0.25)],
          "duree montee")

    # ---- 5. tonneau ----
    sweep("5. TONNEAU (1 tour complet) — duree minimale",
          "   (mu tourne de 360 deg autour du vecteur vitesse)",
          [(f"{d:.2f} s", [("roll", d, {"n_turns": 1.0})], 5.0) for d in
           (2.0, 1.5, 1.0, 0.7, 0.5, 0.35, 0.25)],
          "duree tonneau")

    # ---- 6. vol tranche soutenu ----
    sweep("6. VOL TRANCHE (mu=90 deg) — influence de la vitesse",
          "   (en tranche, l'aile ne porte plus : c'est le fuselage qui travaille)",
          [(f"{v:.1f} m/s", [("knife_edge", 2.0, {"mu_deg": 90.0})], v) for v in
           (2.0, 3.0, 4.0, 5.0, 7.0, 9.0)],
          "vitesse")

    # ---- 7. le '40/60' : quelle repartition tient ? ----
    sweep("7. REPARTITION droit/virage sur 1.0 s (virage 90 deg)",
          "   (moins de temps pour le virage = plus exigeant)",
          [(f"{int(100*w)}% droit", [("straight", 1.0*w, {}),
                                     ("turn", 1.0*(1-w), {"delta_chi_deg": 90.0})], 5.0)
           for w in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9)],
          "repartition")

    print("\n" + "=" * 68)
    print("Ces plages definissent l'espace de recherche du RL.")
    print("=" * 68)