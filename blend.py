"""COMPOSITION PAR MELANGE CONTINU (version "blended").

DIFFERENCE AVEC LA VERSION SEQUENTIELLE
---------------------------------------
  sequentiel : 40 % du temps en ligne droite, PUIS 60 % en virage
                -> les primitives se succedent
  melange    : a CHAQUE INSTANT le mouvement est une combinaison ponderee
                -> a t donne, c'est 77 % "virage" + 23 % "ligne droite"

PRINCIPE
--------
Chaque primitive n'est plus une figure complete mais un TAUX DE CHANGEMENT :

    straight : rien ne change
    turn     : chi_dot   = +/- omega_chi
    climb    : gamma_dot = +/- omega_gamma
    roll     : mu_dot    = +/- omega_mu

A chaque instant on combine :

    chi_dot(t) = somme_i  w_i(t) * taux_chi_i        avec  somme_i w_i(t) = 1

puis on integre pour obtenir gamma(t), chi(t), mu(t), speed(t).

CONSEQUENCE
-----------
Le vol tranche n'a plus besoin d'etre une primitive : il EMERGE. On roule
jusqu'a mu = 90 deg, puis le poids passe sur "straight" qui maintient mu.
Le catalogue devient plus simple ET plus expressif.

LES POIDS DANS LE TEMPS
-----------------------
Les poids sont donnes en quelques POINTS DE CONTROLE et interpoles en douceur
(smoothstep) entre eux. La sortie est donc reellement continue, et les derivees
restent bornees.
"""
import numpy as np

# taux de reference a poids 1 (unites par seconde)
OMEGA_CHI = np.deg2rad(75.0)     # virage : 75 deg/s a plein regime
OMEGA_GAM = np.deg2rad(60.0)     # pente  : 60 deg/s
OMEGA_MU = np.deg2rad(240.0)     # roulis : 240 deg/s
DV = 1.5                         # acceleration : 1.5 m/s^2

# (gamma_dot, chi_dot, mu_dot, speed_dot) par unite de poids
RATES = {
    "straight":   (0.0,        0.0,        0.0,       0.0),
    "turn_right": (0.0,       +OMEGA_CHI,  0.0,       0.0),
    "turn_left":  (0.0,       -OMEGA_CHI,  0.0,       0.0),
    "climb_up":   (+OMEGA_GAM, 0.0,        0.0,       0.0),
    "climb_down": (-OMEGA_GAM, 0.0,        0.0,       0.0),
    "roll_right": (0.0,        0.0,       +OMEGA_MU,  0.0),
    "roll_left":  (0.0,        0.0,       -OMEGA_MU,  0.0),
    "accel":      (0.0,        0.0,        0.0,      +DV),
    "decel":      (0.0,        0.0,        0.0,      -DV),
}
PRIMS = list(RATES.keys())
N_PRIMS = len(PRIMS)


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def weights_over_time(control_weights, t, duration):
    """Interpole les poids entre points de controle (smoothstep), puis normalise.

    control_weights : (K, N_PRIMS) — K points de controle repartis sur la duree
    retourne        : (len(t), N_PRIMS), chaque ligne somme a 1
    """
    C = np.asarray(control_weights, float)
    C = np.clip(C, 0.0, None)
    C = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-9)
    K = len(C)
    if K == 1:
        return np.repeat(C, len(t), axis=0)
    pos = t / max(duration, 1e-9) * (K - 1)      # position continue entre 0 et K-1
    i0 = np.clip(np.floor(pos).astype(int), 0, K - 2)
    u = smoothstep(pos - i0)[:, None]
    W = (1 - u) * C[i0] + u * C[i0 + 1]
    return W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)


def integrate(W, t, state0, gamma_limit_deg=90.0, speed_range=(3.0, 7.0)):
    """Integre les taux ponderes pour obtenir gamma, chi, mu, speed et leurs derivees."""
    R = np.array([RATES[p] for p in PRIMS])          # (N_PRIMS, 4)
    rates = W @ R                                     # (n, 4) : gdot, cdot, mdot, sdot
    n = len(t)
    gamma = np.zeros(n); chi = np.zeros(n); mu = np.zeros(n); speed = np.zeros(n)
    gamma[0] = state0["gamma"]; chi[0] = state0["chi"]
    mu[0] = state0["mu"]; speed[0] = state0["speed"]
    lim = np.deg2rad(gamma_limit_deg)
    gd = rates[:, 0].copy(); cd = rates[:, 1].copy()
    md = rates[:, 2].copy(); sd = rates[:, 3].copy()
    for i in range(n - 1):
        dt = t[i+1] - t[i]
        g_new = np.clip(gamma[i] + gd[i]*dt, -lim, lim)
        s_new = np.clip(speed[i] + sd[i]*dt, speed_range[0], speed_range[1])
        # si la borne mord, le taux effectif est plus faible (coherence des derivees)
        gd[i] = (g_new - gamma[i]) / dt if dt > 0 else 0.0
        sd[i] = (s_new - speed[i]) / dt if dt > 0 else 0.0
        gamma[i+1] = g_new; speed[i+1] = s_new
        chi[i+1] = chi[i] + cd[i]*dt
        mu[i+1] = mu[i] + md[i]*dt
    gd[-1] = gd[-2] if n > 1 else 0.0
    sd[-1] = sd[-2] if n > 1 else 0.0
    return dict(t=t, gamma=gamma, chi=chi, mu=mu, speed=speed,
                gamma_dot=gd, chi_dot=cd, mu_dot=md, speed_dot=sd, weights=W)


def blend(control_weights, duration, state0=None, dt=0.02):
    """Construit une figure par melange continu."""
    state0 = state0 or dict(gamma=0.0, chi=0.0, mu=0.0, speed=5.0)
    t = np.arange(0.0, duration + dt*0.5, dt)
    W = weights_over_time(control_weights, t, duration)
    seq = integrate(W, t, state0)
    seq["marks"] = [(0.0, duration, "blend")]
    return seq


def mix_at(seq, time_s, top=3):
    """Quel melange a l'instant t ? (pour l'explication)"""
    i = int(np.searchsorted(seq["t"], time_s))
    i = min(max(i, 0), len(seq["t"]) - 1)
    w = seq["weights"][i]
    order = np.argsort(-w)[:top]
    return [(PRIMS[j], float(w[j])) for j in order if w[j] > 0.01]