"""Demo : composer des figures F3P a partir de primitives.

    python demo.py

Si le paquet f3p_attitude du superviseur est importable, chaque figure est en
plus validee par SON solveur (poussee, angle d'attaque, convergence).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from compose import initial_state, sequence, mix, check_continuity, to_solver_input

try:
    from f3p_attitude.solver import solve_trajectory
    from f3p_attitude.constants import THRUST_MAX, THRUST_MIN
    HAS_SOLVER = True
except ImportError:
    HAS_SOLVER = False

st = initial_state(speed=5.0)

print("=" * 64)
print("1. LE '40% DROIT, 60% VIRAGE' DU SUPERVISEUR")
print("=" * 64)
for w_straight in (0.2, 0.4, 0.6, 0.8):
    seq = mix(st, 2.0, [("straight", w_straight), ("turn", 1 - w_straight)],
              {"turn": {"delta_chi_deg": 90.0}})
    d = seq["marks"]
    print(f"  {100*w_straight:3.0f}% droit / {100*(1-w_straight):3.0f}% virage  ->  "
          f"droit {d[0][1]-d[0][0]:.2f}s, virage {d[1][1]-d[1][0]:.2f}s, "
          f"|chi_dot| max {np.abs(seq['chi_dot']).max():.2f} rad/s")
print("  -> moins de temps pour virer = virage plus serre = plus exigeant")

print("\n" + "=" * 64)
print("2. FIGURES COMPOSEES")
print("=" * 64)
figures = {
    "montee verticale": [("straight", 1.0, {}), ("climb", 1.5, {"delta_gamma_deg": 90.0}),
                         ("straight", 1.0, {}), ("climb", 1.5, {"delta_gamma_deg": -90.0})],
    "vol tranche + virage": [("straight", 0.8, {}), ("knife_edge", 2.0, {"mu_deg": 90.0}),
                             ("turn", 1.2, {"delta_chi_deg": 180.0})],
    "tonneau en montee": [("climb", 1.2, {"delta_gamma_deg": 45.0}),
                          ("roll", 1.5, {"n_turns": 1.0}),
                          ("climb", 1.2, {"delta_gamma_deg": -45.0})],
}
for nom, steps in figures.items():
    seq = sequence(st, steps)
    chk = check_continuity(seq)
    line = f"  {nom:22s} {seq['t'][-1]:5.1f}s  {len(steps)} primitives  " \
           f"continuite {'OK' if chk['ok'] else 'NON'}"
    if HAS_SOLVER:
        r = solve_trajectory(**to_solver_input(seq))
        th = np.asarray(r.thrust)
        ok = bool(np.asarray(r.success).all() and (th <= THRUST_MAX).all())
        line += f"  | poussee max {th.max():.2f}/{THRUST_MAX:.2f} N  " \
                f"alpha max {np.degrees(np.abs(r.alpha)).max():3.0f} deg  " \
                f"{'VOLABLE' if ok else 'NON VOLABLE'}"
    print(line)

if not HAS_SOLVER:
    print("\n  (f3p_attitude non trouve : lance ce script la ou le paquet du")
    print("   superviseur est importable pour avoir la validation physique)")
