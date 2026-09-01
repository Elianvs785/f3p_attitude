"""DIAGNOSTIC : pourquoi la trajectoire de l'oracle est-elle infaisable ?

On ne devine pas, on ABLATE. On reprend la trajectoire de l'oracle et on la
redonne au solveur en annulant une seule demande a la fois :

    A. baseline           : tel quel
    B. sans acceleration  : speed_dot = 0   (l'avion garde sa vitesse)
    C. sans rotation      : chi_dot = gamma_dot = 0   (il vole droit)
    D. sans les deux      : borne basse, ce que coute le vol lui-meme

Celle qui rend la figure volable designe le coupable. Si aucune ne suffit, c'est
la COMBINAISON (accelerer ET virer en meme temps) qui depasse le budget.

RAPPEL : le juge est le RESIDU (equilibre des forces), pas r.success ni la
poussee (bornee par construction dans le solveur, donc elle ne depasse jamais).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from test_setup import build
from oracle import pursuit_action
from f3p_attitude.solver import solve_trajectory
from f3p_attitude.constants import THRUST_MAX, RESIDUAL_TOL, MASS, G


def run_oracle():
    env, W, T, v_seg, tg, mu = build()
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(pursuit_action(env, k_ang=4.0, k_mu=4.0))
    return env.result(), T


def solve(res, kill_acc=False, kill_rot=False):
    sd = np.zeros_like(res["speed_dot"]) if kill_acc else res["speed_dot"]
    gd = np.zeros_like(res["gamma_dot"]) if kill_rot else res["gamma_dot"]
    cd = np.zeros_like(res["chi_dot"])   if kill_rot else res["chi_dot"]
    r = solve_trajectory(res["t"], res["gamma"], res["chi"], res["speed"],
                         res["mu"], gamma_dot=gd, chi_dot=cd, speed_dot=sd)
    return np.asarray(r.residual_norm, float), np.asarray(r.thrust, float)


if __name__ == "__main__":
    res, T = run_oracle()
    t = res["t"]

    # ---- 1. decomposition de l'acceleration DEMANDEE -------------------
    # a = speed_dot * v_hat + speed * v_hat_dot
    #     ^ longitudinale     ^ transverse (centripete)
    # |v_hat_dot| = sqrt(gamma_dot^2 + (chi_dot*cos gamma)^2)
    a_long = np.abs(res["speed_dot"])
    om = np.sqrt(res["gamma_dot"]**2 + (res["chi_dot"]*np.cos(res["gamma"]))**2)
    a_lat = res["speed"] * om
    a_tot = np.sqrt(a_long**2 + a_lat**2)

    # budget : poussee max / masse, dont il faut retrancher de quoi porter le poids
    a_budget = THRUST_MAX/MASS
    print("="*66)
    print("0. BUDGET D'ACCELERATION")
    print("="*66)
    print(f"  poussee max / masse       : {a_budget:5.1f} m/s2")
    print(f"  dont il faut porter g     : {G:5.1f} m/s2")
    print(f"  reste (ordre de grandeur) : {a_budget-G:5.1f} m/s2")
    print("  (les ailes aident aux vitesses moyennes, pas en vol tres lent)")

    # ---- 2. ablations --------------------------------------------------
    print("\n" + "="*66)
    print("1. ABLATIONS  (juge = residu max, tolerance %.0e)" % RESIDUAL_TOL)
    print("="*66)
    cases = [("A. baseline",          dict()),
             ("B. sans acceleration", dict(kill_acc=True)),
             ("C. sans rotation",     dict(kill_rot=True)),
             ("D. sans les deux",     dict(kill_acc=True, kill_rot=True))]
    out = {}
    print(f"{'cas':<22}{'residu max':>12}{'instants KO':>13}{'poussee max':>13}")
    print("-"*60)
    for name, kw in cases:
        rn, th = solve(res, **kw)
        bad = int((rn > RESIDUAL_TOL).sum())
        out[name] = (rn, bad)
        print(f"{name:<22}{rn.max():>12.2e}{bad:>8}/{len(rn):<4}"
              f"{100*th.max()/THRUST_MAX:>11.0f}%")

    # ---- 3. ou et combien ----------------------------------------------
    rn0 = out["A. baseline"][0]
    bad = np.where(rn0 > RESIDUAL_TOL)[0]
    print("\n" + "="*66)
    print("2. LES INSTANTS FAUTIFS")
    print("="*66)
    if len(bad) == 0:
        print("  aucun.")
    else:
        print(f"{'t':>7}{'v':>8}{'a_long':>9}{'a_lat':>9}{'a_tot':>9}{'residu':>11}")
        print("-"*54)
        for i in bad:
            print(f"{t[i]:>7.2f}{res['speed'][i]:>8.2f}{a_long[i]:>9.1f}"
                  f"{a_lat[i]:>9.1f}{a_tot[i]:>9.1f}{rn0[i]:>11.2e}")
        print(f"\n  fenetre : t = {t[bad].min():.2f} -> {t[bad].max():.2f} s")
        print(f"  waypoints encadrants : "
              f"{T[T <= t[bad].min()][-1]:.1f} s et {T[T >= t[bad].max()][0]:.1f} s")

    # ---- 4. ordre de grandeur global -----------------------------------
    print("\n" + "="*66)
    print("3. DEMANDE MAXIMALE SUR TOUTE LA FIGURE")
    print("="*66)
    for nm, arr in [("longitudinale", a_long), ("transverse", a_lat),
                    ("totale", a_tot)]:
        print(f"  a_{nm:<15} max {arr.max():>6.1f} m/s2   "
              f"(a t = {t[arr.argmax()]:.2f} s)")
    print(f"\n  budget disponible ~ {a_budget-G:.1f} m/s2 en vol lent, "
          f"{a_budget:.1f} m/s2 au mieux.")
