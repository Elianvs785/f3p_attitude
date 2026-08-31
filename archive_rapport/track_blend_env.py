"""SUIVI DE TRAJECTOIRE par melange continu.

L'IDEE
------
Au lieu de demander au RL d'inventer une figure, on lui donne une TRAJECTOIRE
CIBLE et il doit trouver, a chaque instant, le bon melange de primitives pour la
reproduire.

POURQUOI C'EST MIEUX QUE LA VERSION "INVENTER UNE FIGURE"
--------------------------------------------------------
1. Le melange DOIT varier dans le temps (tourner ici, monter la, ralentir avant
   ce virage). Dans la version precedente, la politique convergeait vers un
   melange CONSTANT (variation = 0.00) car rien ne l'obligeait a changer.
2. La recompense devient DENSE : une erreur de suivi a chaque pas, au lieu d'un
   score unique en fin d'episode. Le credit est bien mieux attribue.
3. Ca relie les deux parties du projet : le generateur minimum snap produit une
   trajectoire optimale, le RL apprend a la voler avec un vrai avion F3P.

FORMULATION
-----------
  etat   : etat de vol courant (gamma, chi, mu, speed)
           + ERREUR par rapport a la reference
           + APERCU de la reference a venir (lookahead)
  action : vecteur reel de dimension N_PRIMS -> softmax -> poids du melange
  reward : -(erreur de suivi) a chaque pas, + bonus de faisabilite
"""
import numpy as np
from blend import RATES, PRIMS, N_PRIMS, smoothstep

W_GAMMA, W_CHI, W_SPEED = 1.0, 1.0, 0.35     # ponderation des erreurs
LOOKAHEAD = 3                                  # nb de pas de reference vus a l'avance


def softmax(a):
    a = np.asarray(a, float); a = a - a.max()
    e = np.exp(a); return e / max(e.sum(), 1e-12)


def reference_from_weights(control_weights, duration, dt, state0):
    """Genere une reference ATTEIGNABLE PAR CONSTRUCTION (oracle).

    Sert a valider que l'environnement permet un suivi parfait avant de demander
    au RL d'apprendre. Si l'oracle ne suit pas, le probleme vient de
    l'environnement, pas de l'apprentissage.
    """
    from blend import blend
    return blend(control_weights, duration, state0, dt)


class TrackEnv:
    """Suivi d'une reference (gamma, chi, speed) par melange de primitives."""

    def __init__(self, dt_ctrl=0.25, dt=0.02, speed_range=(3.0, 7.0),
                 gamma_limit_deg=90.0):
        self.dt_ctrl = dt_ctrl        # duree pendant laquelle un melange est applique
        self.dt = dt
        self.speed_range = speed_range
        self.gamma_lim = np.deg2rad(gamma_limit_deg)
        self.R = np.array([RATES[p] for p in PRIMS])     # (N_PRIMS, 4)
        self.action_dim = N_PRIMS
        self.state_dim = 4 + 3 + 3*LOOKAHEAD

    # ------------------------------------------------------------------
    def set_reference(self, ref):
        """ref : dict avec t, gamma, chi, speed (issu d'un blend ou du minimum snap)."""
        self.ref = ref
        self.T = float(ref["t"][-1])
        self.n_steps = int(round(self.T / self.dt_ctrl))

    def _ref_at(self, time_s):
        t = self.ref["t"]
        i = int(np.clip(np.searchsorted(t, time_s), 0, len(t)-1))
        return (float(self.ref["gamma"][i]), float(self.ref["chi"][i]),
                float(self.ref["speed"][i]))

    def reset(self):
        g0, c0, s0 = self._ref_at(0.0)
        self.state = dict(gamma=g0, chi=c0, mu=0.0, speed=s0)
        self.k = 0
        self.traj = dict(t=[0.0], gamma=[g0], chi=[c0], mu=[0.0], speed=[s0],
                         gamma_dot=[0.0], chi_dot=[0.0], mu_dot=[0.0], speed_dot=[0.0],
                         weights=[])
        return self._obs()

    def _obs(self):
        s = self.state
        tnow = self.k * self.dt_ctrl
        gr, cr, sr = self._ref_at(tnow)
        obs = [s["gamma"]/(np.pi/2), np.sin(s["chi"]), np.cos(s["chi"]),
               (s["speed"]-5.0)/2.0,
               (s["gamma"]-gr)/(np.pi/2),                 # erreur de pente
               np.arctan2(np.sin(s["chi"]-cr), np.cos(s["chi"]-cr))/np.pi,  # erreur de cap
               (s["speed"]-sr)/2.0]                        # erreur de vitesse
        for j in range(1, LOOKAHEAD+1):                    # apercu de la suite
            gr2, cr2, sr2 = self._ref_at(tnow + j*self.dt_ctrl)
            obs += [(gr2-s["gamma"])/(np.pi/2),
                    np.arctan2(np.sin(cr2-s["chi"]), np.cos(cr2-s["chi"]))/np.pi,
                    (sr2-s["speed"])/2.0]
        return np.array(obs, float)

    def step(self, action):
        """Applique le melange pendant dt_ctrl, puis mesure l'erreur de suivi."""
        w = softmax(action)
        rates = w @ self.R                       # (gamma_dot, chi_dot, mu_dot, speed_dot)
        n_sub = max(int(round(self.dt_ctrl/self.dt)), 1)
        s = self.state
        for _ in range(n_sub):
            g = float(np.clip(s["gamma"] + rates[0]*self.dt, -self.gamma_lim, self.gamma_lim))
            sp = float(np.clip(s["speed"] + rates[3]*self.dt,
                               self.speed_range[0], self.speed_range[1]))
            gd = (g - s["gamma"])/self.dt
            sd = (sp - s["speed"])/self.dt
            s = dict(gamma=g, chi=s["chi"] + rates[1]*self.dt,
                     mu=s["mu"] + rates[2]*self.dt, speed=sp)
            self.traj["t"].append(self.traj["t"][-1] + self.dt)
            for k_, v_ in [("gamma",g), ("chi",s["chi"]), ("mu",s["mu"]), ("speed",sp),
                           ("gamma_dot",gd), ("chi_dot",rates[1]),
                           ("mu_dot",rates[2]), ("speed_dot",sd)]:
                self.traj[k_].append(v_)
        self.state = s
        self.traj["weights"].append(w)
        self.k += 1

        # --- erreur de suivi ---
        gr, cr, sr = self._ref_at(self.k*self.dt_ctrl)
        e_g = abs(s["gamma"] - gr)
        e_c = abs(np.arctan2(np.sin(s["chi"]-cr), np.cos(s["chi"]-cr)))
        e_s = abs(s["speed"] - sr)
        err = W_GAMMA*e_g + W_CHI*e_c + W_SPEED*e_s
        reward = float(np.exp(-2.0*err))          # 1 si parfait, ->0 si loin
        done = self.k >= self.n_steps
        return self._obs(), reward, done, dict(e_gamma=e_g, e_chi=e_c, e_speed=e_s)

    def result(self):
        out = {k: np.asarray(v) for k, v in self.traj.items() if k != "weights"}
        out["weights"] = np.asarray(self.traj["weights"])
        out["marks"] = [(0.0, float(out["t"][-1]), "track")]
        return out

    def tracking_error(self):
        """Erreur moyenne sur toute la trajectoire (metrique de qualite)."""
        r = self.result()
        eg, ec, es = [], [], []
        for i, ti in enumerate(r["t"]):
            gr, cr, sr = self._ref_at(float(ti))
            eg.append(abs(r["gamma"][i]-gr))
            ec.append(abs(np.arctan2(np.sin(r["chi"][i]-cr), np.cos(r["chi"][i]-cr))))
            es.append(abs(r["speed"][i]-sr))
        return dict(gamma_deg=float(np.degrees(np.mean(eg))),
                    chi_deg=float(np.degrees(np.mean(ec))),
                    speed=float(np.mean(es)))