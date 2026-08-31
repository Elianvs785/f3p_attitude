"""Environnement RL CONTINU pour la composition par melange.

ACTION CONTINUE
---------------
A chaque point de controle, la politique sort un vecteur de R^N_PRIMS.
Les poids du melange sont obtenus par softmax :

    w = softmax(a)        avec  a ~ N(mu_theta(s), sigma_theta)

C'est ce qui rend l'action CONTINUE : le RL ne choisit plus une primitive dans
un catalogue, il dose librement le melange. Le softmax garantit w >= 0 et
somme(w) = 1 sans contrainte explicite.

RECOMPENSE
----------
    figure involable                     -> -1
    sinon  0.35 * poussee (cible 85 %)
         + 0.25 * diversite  (entropie du melange moyen)
         + 0.20 * variation  (le melange evolue au lieu de rester fige)
         + 0.20 * amplitude  (angles parcourus, chaque axe normalise)

La DIVERSITE est mesuree par l'entropie de la distribution moyenne des poids :
elle est maximale quand toutes les primitives sont utilisees, minimale quand une
seule domine. C'est l'equivalent continu du "ne pas repeter la meme primitive".
"""
import numpy as np
from blend import blend, PRIMS, N_PRIMS

STATE_DIM = 6


def softmax(a):
    a = np.asarray(a, float)
    a = a - a.max()
    e = np.exp(a)
    return e / max(e.sum(), 1e-12)


class BlendEnv:
    def __init__(self, n_control=5, duration=5.0, dt=0.04,
                 speed0=5.0, box=(40.0, 40.0, 10.0),
                 thrust_target=0.85, residual_tol=1e-4,
                 alpha_stall=25.0, roll_max_deg=720.0):
        self.K = n_control; self.duration = duration; self.dt = dt
        self.speed0 = speed0; self.box = box
        self.thrust_target = thrust_target; self.residual_tol = residual_tol
        self.alpha_stall = alpha_stall; self.roll_max = roll_max_deg
        self.action_dim = N_PRIMS
        self.state_dim = STATE_DIM

    # -------------------------------------------------- deroulement
    def reset(self):
        self.k = 0
        self.controls = []
        self.state = dict(gamma=0.0, chi=0.0, mu=0.0, speed=self.speed0)
        return self._obs()

    def _obs(self):
        s = self.state
        return np.array([s["gamma"]/(np.pi/2), np.sin(s["chi"]), np.cos(s["chi"]),
                         s["mu"]/np.pi, (s["speed"]-5.0)/2.0,
                         self.k/max(self.K-1, 1)], float)

    def step(self, action):
        """action : vecteur reel de dimension N_PRIMS (avant softmax)."""
        self.controls.append(softmax(action))
        self.k += 1
        done = self.k >= self.K
        if not done:
            # etat intermediaire : on integre ce qu'on a jusqu'ici
            seq = blend(np.array(self.controls), self.duration*self.k/self.K,
                        dict(gamma=0.0, chi=0.0, mu=0.0, speed=self.speed0), self.dt)
            self.state = dict(gamma=float(seq["gamma"][-1]), chi=float(seq["chi"][-1]),
                              mu=float(seq["mu"][-1]), speed=float(seq["speed"][-1]))
        return self._obs(), done

    def build(self):
        return blend(np.array(self.controls), self.duration,
                     dict(gamma=0.0, chi=0.0, mu=0.0, speed=self.speed0), self.dt)

    # -------------------------------------------------- evaluation
    def evaluate(self, seq, solver):
        W = seq["weights"]

        # contrainte de volume (le F3P se vole en salle)
        t, g, c, s = seq["t"], seq["gamma"], seq["chi"], seq["speed"]
        v = np.stack([s*np.cos(g)*np.cos(c), s*np.cos(g)*np.sin(c), s*np.sin(g)], 1)
        P = np.zeros_like(v)
        for i in range(len(t)-1):
            P[i+1] = P[i] + 0.5*(v[i]+v[i+1])*(t[i+1]-t[i])
        span = P.max(0) - P.min(0)
        if (span > np.array(self.box)).any():
            return -1.0, dict(feasible=False, why="hors salle", span=tuple(np.round(span,1)))

        roll = float(np.degrees(np.abs(seq["mu_dot"]).max()))
        if roll > self.roll_max:
            return -1.0, dict(feasible=False, why="roulis", roll=roll)

        r = solver(seq)
        if r is None:
            return -1.0, dict(feasible=False, why="solveur")
        thrust, alpha, res = r
        if res > self.residual_tol:
            return -1.0, dict(feasible=False, why="equilibre", thrust=thrust, res=res)
        if alpha > self.alpha_stall:
            return -1.0, dict(feasible=False, why="decrochage", alpha=alpha)

        # --- notes ---
        s_thrust = float(np.clip(1 - abs(thrust - self.thrust_target)/self.thrust_target, 0, 1))
        wbar = W.mean(axis=0)
        ent = -(wbar * np.log(wbar + 1e-12)).sum() / np.log(N_PRIMS)   # 0..1
        s_div = float(ent)
        s_var = float(np.clip(np.abs(np.diff(W, axis=0)).sum() / (0.5*len(W)), 0, 1))
        a_chi = np.clip(np.abs(np.diff(seq["chi"])).sum()/(2*np.pi), 0, 1)
        a_gam = np.clip(np.abs(np.diff(seq["gamma"])).sum()/np.pi, 0, 1)
        a_mu = np.clip(np.abs(np.diff(seq["mu"])).sum()/(2*np.pi), 0, 1)
        s_amp = float((a_chi + a_gam + a_mu)/3)

        reward = 0.35*s_thrust + 0.25*s_div + 0.20*s_var + 0.20*s_amp
        return float(reward), dict(feasible=True, thrust=thrust, alpha=alpha, res=res,
                                   roll=roll, s_thrust=s_thrust, s_div=s_div,
                                   s_var=s_var, s_amp=s_amp,
                                   span=tuple(np.round(span,1)),
                                   wbar={PRIMS[i]: round(float(wbar[i]),3)
                                         for i in np.argsort(-wbar)[:4]})