"""ORACLE : poursuite pure — la borne basse a battre.

Regle deterministe : viser le prochain waypoint, en deduire les taux de rotation
necessaires, et construire le melange de primitives qui les produit.

CONSTRUCTION DIRECTE DES POIDS
------------------------------
Une premiere version notait chaque primitive par sa projection sur le taux
souhaite, puis appliquait un softmax. Mauvais : le taux resultant est une moyenne
ponderee, qui ne vaut pas le taux voulu (mesure : 3.2 m d'ecart).

Ici on construit les poids directement. Chaque axe a une primitive + et une -, et
le taux obtenu est proportionnel au poids :

    poids sur turn_right = |taux_chi_voulu| / OMEGA_CHI      (si taux > 0)

Si la somme depasse 1, on normalise (l'avion fait de son mieux) ; sinon le reste
va sur "straight".

Sert a repondre a "l'environnement PERMET-il le suivi ?" avant d'entrainer.
"""
import numpy as np
from track_f3p_env import PRIMS, N_PRIMS, OMEGA_CHI, OMEGA_GAM, OMEGA_MU

I = {p: PRIMS.index(p) for p in PRIMS}


def pursuit_weights(env, k_ang=4.0, k_mu=4.0, lead=0.0):
    """Poids du melange pour viser le prochain waypoint."""
    i = env._target_index(env.t + lead)
    to = env.W[i] - env.pos
    d_des = to/(np.linalg.norm(to) + 1e-9)

    gam_des = np.arcsin(np.clip(d_des[2], -1, 1))
    chi_des = np.arctan2(d_des[1], d_des[0])

    e_gam = gam_des - env.gamma
    e_chi = np.arctan2(np.sin(chi_des - env.chi), np.cos(chi_des - env.chi))
    dmu = env.mu_ref_fn(env.t) - env.mu
    e_mu = np.arctan2(np.sin(dmu), np.cos(dmu))

    # taux souhaites : proportionnels a l'ecart (correcteur proportionnel)
    want_gam = k_ang*e_gam
    want_chi = k_ang*e_chi
    want_mu = k_mu*e_mu

    w = np.zeros(N_PRIMS)
    w[I["pitch_up"]   if want_gam > 0 else I["pitch_down"]] = abs(want_gam)/OMEGA_GAM
    w[I["turn_right"] if want_chi > 0 else I["turn_left"]]  = abs(want_chi)/OMEGA_CHI
    w[I["roll_right"] if want_mu  > 0 else I["roll_left"]]  = abs(want_mu)/OMEGA_MU

    tot = w.sum()
    if tot > 1.0:
        w /= tot                      # saturation : l'avion fait de son mieux
    else:
        w[I["straight"]] = 1.0 - tot  # le reste : ne rien changer
    return w


def pursuit_rates(env, k_ang=4.0, k_mu=4.0, lead=0.0):
    """Taux souhaites, en fraction du maximum, dans [-1, 1]."""
    i = env._target_index(env.t + lead)
    to = env.W[i] - env.pos
    d_des = to/(np.linalg.norm(to) + 1e-9)
    gam_des = np.arcsin(np.clip(d_des[2], -1, 1))
    chi_des = np.arctan2(d_des[1], d_des[0])
    e_gam = gam_des - env.gamma
    e_chi = np.arctan2(np.sin(chi_des - env.chi), np.cos(chi_des - env.chi))
    dmu = env.mu_ref_fn(env.t) - env.mu
    e_mu = np.arctan2(np.sin(dmu), np.cos(dmu))
    return np.array([k_ang*e_gam/OMEGA_GAM,
                     k_ang*e_chi/OMEGA_CHI,
                     k_mu*e_mu/OMEGA_MU])


def pursuit_action(env, speed_mod=0.0, **kw):
    """Action au format attendu par env.action_mode.

    "rates"   : 3 nombres pre-tanh + modulation de vitesse
    "softmax" : les log-poids (ancien format)
    """
    if getattr(env, "action_mode", "softmax") == "rates":
        r = np.clip(pursuit_rates(env, **kw), -0.999, 0.999)
        return np.concatenate([np.arctanh(r), [speed_mod]])
    w = pursuit_weights(env, **kw)
    return np.concatenate([np.log(np.maximum(w, 1e-9)), [speed_mod]])