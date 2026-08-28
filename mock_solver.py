"""Solveur SIMULE — uniquement pour valider la mecanique du RL sans f3p_attitude.

Reproduit grossierement les limites mesurees par explore_limits.py :
  poussee ~ f(vitesse, taux de virage, pente)
  decrochage si la vitesse est trop faible
Le vrai entrainement utilise le solveur du superviseur.
"""
import numpy as np

def mock(seq):
    sp = seq["speed"]; gd = np.abs(seq["gamma_dot"]); cd = np.abs(seq["chi_dot"])
    g = seq["gamma"]
    # poussee : trainee (v^2) + montee (sin gamma) + virage (v * chi_dot)
    th = 0.10 + 0.012*sp**2 + 0.45*np.maximum(np.sin(g),0) + 0.06*sp*cd + 0.05*sp*gd
    thrust = float(th.max()/1.0)          # normalise sur 1.0 = poussee max
    alpha = float(np.degrees(np.arctan2(1.0, np.maximum(sp,0.1)**2*0.25)).max())
    res = 0.0 if thrust <= 1.0 else float(thrust-1.0)
    return thrust, alpha, res
