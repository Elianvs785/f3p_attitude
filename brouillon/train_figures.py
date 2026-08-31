"""Entrainement du RL de composition de figures F3P — avec le VRAI solveur.

    python train_figures.py

Le solveur du superviseur evalue chaque figure (~0.2 s), donc ~400 episodes
prennent quelques minutes. Si f3p_attitude n'est pas trouve, un solveur simule
prend le relais (utile pour verifier la mecanique, pas pour des resultats).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from figure_rl import train, rollout, CATALOG, N_MOVES, build
from compose import to_solver_input

EPISODES = 400
N_STEPS = 6            # nombre de primitives par figure

try:
    from f3p_attitude.solver import solve_trajectory
    from f3p_attitude.constants import THRUST_MAX
    def solver(seq):
        try:
            r = solve_trajectory(**to_solver_input(seq))
        except Exception:
            return None
        th = float(np.asarray(r.thrust).max() / THRUST_MAX)
        al = float(np.degrees(np.abs(np.asarray(r.alpha))).max())
        res = float(np.asarray(r.residual_norm).max())
        return th, al, res
    print("solveur : f3p_attitude (physique reelle)")
except ImportError:
    from mock_solver import mock as solver
    print("solveur : SIMULE (f3p_attitude introuvable) — resultats indicatifs")

if __name__ == "__main__":
    print(f"catalogue : {N_MOVES} mouvements | figures de {N_STEPS} primitives\n")
    t0 = time.time()
    pol, hist, feas = train(solver, episodes=EPISODES, n_steps=N_STEPS,
                            lr=0.05, seed=2, log_every=50)
    print(f"\nentraine en {time.time()-t0:.0f}s")
    print(f"  reward   : {np.mean(hist[:50]):+.3f} -> {np.mean(hist[-50:]):+.3f}")
    print(f"  volables : {100*np.mean(feas[:50]):.0f}% -> {100*np.mean(feas[-50:]):.0f}%")

    print("\n--- meilleures figures apprises ---")
    rng = np.random.default_rng(0)
    best = []
    for _ in range(30):
        R, _, mv, sq, info = rollout(pol, rng, N_STEPS, solver)
        if info["feasible"]:
            best.append((R, mv, info))
    best.sort(key=lambda x: -x[0])
    for rank, (R, moves, info) in enumerate(best[:3], 1):
        print(f"\n  #{rank}  reward {R:+.3f} | poussee {100*info['thrust']:.0f}% "
              f"| diversite {info['s_div']:.2f} | non-repet {info['s_rep']:.2f} "
              f"(consec {info.get('s_consec',0):.2f}, equilibre {info.get('s_balance',0):.2f})")
        for i, (idx, dur) in enumerate(moves, 1):
            n, kw, sp = CATALOG[idx]
            print(f"      {i}. {n:11s} {dur:.1f}s  {kw}")
    np.save("training_reward.npy", np.array(hist))
    np.save("training_feasible.npy", np.array(feas))