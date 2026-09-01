"""SUIVI MUSICAL v2 — reconstruit sur les fondations de l'env qui MARCHAIT.

POURQUOI UNE V2 (et pas un 5e correctif de track_f3p_env)
---------------------------------------------------------
L'ancien track_blend_env suivait des references d'ANGLES : erreurs bornees,
observations bornees, recompense dense a gradient partout -> le PPO apprenait
(huit suivi a 0.9/6.8 deg). track_f3p_env suivait des POSITIONS : erreurs non
bornees, et chacun des 4 correctifs (exponentielle eteinte, potentiel, slog,
softmax) rattrapait une consequence de ce choix.

La reduction qui reconcilie les deux est celle de la poursuite : VISER UN POINT
transforme une erreur de position (non bornee) en erreur de direction (bornee).
Cette v2 = fondations de track_blend_env + trois ajouts :
  1. le point de visee issu des waypoints (u - d, borne par 2)
  2. la ponctualite aux beats (recompense a l'instant EXACT du beat)
  3. la vitesse LIBRE + limites de taux DEPENDANTES de la vitesse (enveloppe
     mesuree par explore_limits) — c'est la couche suivi qui porte la physique.

On GARDE de la v1 ce qui a un theoreme ou une mesure pour lui : le potentiel
(Ng, Harada & Russell 1999), le temps calcule par compteur (jamais accumule),
les taux directs (weights_from_rates), slog pour les grandeurs non bornees.

CONTRAT (CONTRAT.md) : la musique contraint des instants discrets ; entre deux
waypoints la repartition de vitesse est libre. Aucun profil v_ref ici.
"""
import numpy as np
import os, sys
_LIS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LISdrone")
if os.path.isdir(_LIS) and _LIS not in sys.path:
    sys.path.insert(0, _LIS)
try:
    from minsnap import evaluate as _ms_eval
    from minsnap_v2 import waypoint_velocities as _ms_wv, solve_v as _ms_solve
    HAS_MINSNAP = True
except Exception:
    HAS_MINSNAP = False

# --------------------------------------------------------------- primitives
PRIMS = ["straight", "turn_right", "turn_left", "pitch_up", "pitch_down",
         "roll_right", "roll_left"]
N_PRIMS = len(PRIMS)
I = {p: i for i, p in enumerate(PRIMS)}

# ---- enveloppe MESUREE (explore_limits.py), plus une seule constante magique.
# virage : 90 deg en 0.5 s a 5 m/s = OK (3.14 rad/s), a 7 m/s = NON.
#          -> force laterale bornee : omega_chi_max(v) = min(OMEGA_ABS, A_LAT/v)
#             A_LAT = 5.0 * 3.14 = 15.7 m/s2
# pente  : verticale en 0.7 s a 5 m/s = OK (2.24 rad/s) -> A_VERT = 11.2 m/s2
# roulis : gratuit dans le modele (pas de moments) -> borne a la main 720 deg/s
#          (limite connue, declaree au rapport)
OMEGA_ABS = np.deg2rad(220.0)      # butee actionneur basse vitesse
A_LAT  = 15.7                      # m/s2
A_VERT = 11.2                      # m/s2
OMEGA_MU = np.deg2rad(720.0)

V_MIN, V_MAX = 0.5, 7.0            # le harrier autorise le tres lent
ACC_MAX = 5.0                      # la vitesse GLISSE vers la consigne
LOOKAHEAD = 2

# ---- recompense : TROIS idees (CONTRAT.md), erreurs toutes bornees
# MESURE (M10) : avec W_TIME=3, la ponctualite ne pesait que 8 % du budget de
# recompense d'un episode — elle tombe 7 fois quand les autres termes tombent
# 150 fois. Comparer des COEFFICIENTS sans regarder les FREQUENCES est une
# erreur d'unites (par-evenement vs par-pas). L'agent optimisait donc chemin
# (40 %) et roulis (40 %) : il volait la bonne trajectoire au mauvais moment.
W_TIME = 20.0       # ponctualite aux beats — reellement dominant (~50 %)
W_MU   = 0.25       # fidelite du roulis (facile a satisfaire : ne pas diluer)
W_SCHED = 1.0       # potentiel d'HORAIRE (voir ci-dessous)
W_REF   = 2.0       # suivi du point mobile — le terme dominant du mode 'reference'
D_TIME = 1.0        # echelle (m) de la ponctualite : exp(-err/1m)

# 7 evenements sur 150 pas restent SPARSES : avec gamma=0.97 le credit ne
# remonte qu'une trentaine de pas, et l'ecart entre les beats a t=2 s et t=10 s
# en fait 80. On ajoute donc un potentiel d'horaire
#     Phi_sched = -|avance parcourue - avance prevue|
# qui donne un gradient de synchronisation a CHAQUE pas. Meme theoreme que le
# potentiel de position : il guide sans changer la politique optimale.


def slog(x):
    x = np.asarray(x, float)
    return np.sign(x)*np.log1p(np.abs(x))


def omega_max(speed):
    """Taux max (gamma, chi) a cette vitesse — l'enveloppe, pas une constante."""
    v = max(float(speed), 1e-3)
    return min(OMEGA_ABS, A_VERT/v), min(OMEGA_ABS, A_LAT/v)


def weights_from_rates(a3, om_gam, om_chi):
    """(a_gam, a_chi, a_mu) dans [-1,1] -> poids des 7 primitives.

    Le poids demande une FRACTION du taux max COURANT (qui depend de la
    vitesse). Parametrisation lineaire — leçon M7/M8 : pas de softmax.
    """
    g, c, m = np.tanh(np.asarray(a3, float)[:3])
    w = np.zeros(N_PRIMS)
    w[I["pitch_up"   if g > 0 else "pitch_down"]] = abs(g)
    w[I["turn_right" if c > 0 else "turn_left"]]  = abs(c)
    w[I["roll_right" if m > 0 else "roll_left"]]  = abs(m)
    tot = w.sum()
    if tot > 1.0: w /= tot
    else: w[I["straight"]] = 1.0 - tot
    # taux effectifs (signes)
    gd = np.sign(g)*(w[I["pitch_up"]]+w[I["pitch_down"]])*om_gam
    cd = np.sign(c)*(w[I["turn_right"]]+w[I["turn_left"]])*om_chi
    md = np.sign(m)*(w[I["roll_right"]]+w[I["roll_left"]])*OMEGA_MU
    return w, gd, cd, md


class TrackMusic:
    """Suivi de waypoints musicaux par melange continu, vitesse libre."""

    def __init__(self, dt_ctrl=0.1, dt=0.02, gamma_limit_deg=89.0):
        self.dt_ctrl, self.dt = dt_ctrl, dt
        self.gam_lim = np.deg2rad(gamma_limit_deg)
        self.t_lead = 0.35                  # anticipation (s) sur le point mobile
        self.action_dim = 4                 # a_gam, a_chi, a_mu, a_speed
        # 6 vol + 3 (u-d) + 3 slog(lat) + 2 virage a venir + 3*LOOKAHEAD + 3 timing
        # 6 vol + 3 visee + 3 erreur + 2 virage + 3*LOOKAHEAD + 8 attitude exigee + 4
        self.state_dim = 6 + 3 + 3 + 2 + 3*LOOKAHEAD + 8 + 4

    # ------------------------------------------------------------- chemin
    def set_path(self, W, T, mu_ref_fn, reference="polyline"):
        """reference="polyline" : point mobile sur la ligne brisee (C0).
        reference="minsnap"  : x(t) minimum snap par les MEMES waypoints aux
        MEMES beats (C2 : vitesse et acceleration continues et FINIES).

        Pourquoi c'est decisif (M16) : la ligne brisee exige une acceleration
        infinie aux coins — AUCUNE politique ne peut la suivre exactement, le
        RL cherche un compromis. Le minsnap est ATTEIGNABLE : le RL cherche une
        solution qui existe, comme dans l'ancien env qui apprenait bien.
        Genere par LISdrone/minsnap_v2 : la generation reste 100 % musicale
        (memes waypoints, memes instants), seule la CIBLE DE SUIVI est lissee.
        """
        self.W = np.asarray(W, float); self.T = np.asarray(T, float)
        self.reference = reference
        if reference == "minsnap":
            if not HAS_MINSNAP:
                raise ImportError("minsnap introuvable : LISdrone doit etre le "
                                  "dossier frere de f3p_attitude")
            self._coeffs = _ms_solve(self.W, self.T, _ms_wv(self.W, self.T))
        self.mu_ref_fn = mu_ref_fn
        self.n_seg = len(W) - 1
        self.seg_len = np.linalg.norm(np.diff(self.W, axis=0), axis=1)
        self.cum_len = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.n_steps = int(round((T[-1]-T[0])/self.dt_ctrl))
        # angle du virage a CHAQUE waypoint interieur (pour l'observation)
        self.turn_ang = np.zeros(len(W))
        for i in range(1, len(W)-1):
            a = self.W[i]-self.W[i-1]; b = self.W[i+1]-self.W[i]
            a /= np.linalg.norm(a)+1e-9; b /= np.linalg.norm(b)+1e-9
            self.turn_ang[i] = np.arccos(np.clip(a@b, -1, 1))

    def _target_index(self, t):
        return int(np.clip(np.searchsorted(self.T, t, side="right"),
                           1, len(self.T)-1))

    def _arclen(self, p):
        best_s, best_d = 0.0, np.inf
        for j in range(self.n_seg):
            a, ab = self.W[j], self.W[j+1]-self.W[j]
            u = float(np.clip((p-a)@ab/(float(ab@ab)+1e-12), 0.0, 1.0))
            d = float(np.linalg.norm(p-(a+u*ab)))
            if d < best_d: best_d, best_s = d, float(self.cum_len[j]+u*self.seg_len[j])
        return best_s, best_d

    # -------------------------------------------------------------- cycle
    def reset(self):
        self.k = 0
        self.t = float(self.T[0])
        self.pos = self.W[0].copy()
        d0 = self.W[1]-self.W[0]; d0 /= np.linalg.norm(d0)+1e-9
        self.gamma = float(np.arcsin(np.clip(d0[2], -1, 1)))
        self.chi = float(np.arctan2(d0[1], d0[0]))
        self.mu = float(self.mu_ref_fn(self.t))
        self.speed = float(self.seg_len[0]/(self.T[1]-self.T[0]))
        rd0, _ = self._ref_state(self.t)
        self.prev_lag = abs(float((self.pos - self._ref_point(self.t)) @ rd0))
        self.log = dict(pos=[self.pos.copy()], t=[self.t], speed=[self.speed],
                        gamma=[self.gamma], chi=[self.chi], mu=[self.mu],
                        chi_dot=[0.0], gamma_dot=[0.0], mu_dot=[0.0])
        self.log_w = []
        self.beat_err = []
        self.track_err = []                    # ||pos(T_i) - W_i|| a l'instant du beat
        return self._obs()

    def _ref_state(self, t, h=0.05):
        """Attitude EXIGEE par la reference a l'instant t : direction + vitesse.

        C'est l'information qui faisait marcher l'ancien env (track_blend_env) :
        l'agent y voyait gamma_ref, chi_ref, speed_ref, maintenant et plus tard,
        et n'avait qu'a les recopier. Mon env ne donnait que des POSITIONS :
        l'agent devait deviner quelle attitude les produit, alors que
        l'information existe. Transporter l'information, ne pas la reconstruire.
        """
        a, b = self._ref_point(t), self._ref_point(t+h)
        d = b - a
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return self._dir(), 0.0
        return d/n, n/h

    def _ref_point(self, t):
        """Point MOBILE de la trajectoire de reference a l'instant t.

        Il glisse le long des lignes droites et se trouve exactement sur W[i] a
        l'instant T[i]. C'est la formulation demandee par le superviseur :
        le RL SUIT LE TRAJET, il ne rejoint pas des points.

        Consequences, toutes mesurables :
          - on ne peut pas DEPASSER une cible qui avance devant soi ;
          - position et horaire cessent d'etre concurrents : ||pos - r(t)||
            contient les deux, puisque etre au bon endroit au bon moment EST la
            meme chose ;
          - la recompense se reduit a un terme dominant + le roulis.
        """
        t = float(np.clip(t, self.T[0], self.T[-1]))
        if self.reference == "minsnap":
            return _ms_eval(self._coeffs, self.T, np.array([t]), 0)[0]
        s = self._s_ref(t)
        j = int(np.clip(np.searchsorted(self.cum_len, s, side="right")-1,
                        0, self.n_seg-1))
        u = (s - self.cum_len[j])/max(self.seg_len[j], 1e-9)
        return self.W[j] + np.clip(u, 0.0, 1.0)*(self.W[j+1]-self.W[j])

    def _s_ref(self, t):
        """Avance PREVUE le long du chemin a l'instant t (horaire des beats)."""
        return float(np.interp(t, self.T, self.cum_len))

    def _dir(self):
        return np.array([np.cos(self.gamma)*np.cos(self.chi),
                         np.cos(self.gamma)*np.sin(self.chi),
                         np.sin(self.gamma)])

    def _obs(self):
        i = self._target_index(self.t)
        # on vise LEGEREMENT DEVANT le point mobile (pure pursuit) : viser le
        # point courant ferait toujours arriver en retard d'un pas.
        to = self._ref_point(self.t + self.t_lead) - self.pos
        dist = float(np.linalg.norm(to))
        u = to/max(dist, 1e-9)
        d = self._dir()
        _, dlat = self._arclen(self.pos)
        proj_dir = None
        t_left = max(float(self.T[i]-self.t), 1e-3)
        # vitesse moyenne REQUISE pour etre a l'heure : dist/t_left. C'est LA
        # grandeur de decision de la vitesse — l'agent doit la voir (bornee).
        v_req = np.clip(dist/t_left, 0.0, 8.0)/8.0
        obs = [self.gamma/(np.pi/2), np.sin(self.chi), np.cos(self.chi),
               (self.speed-3.5)/3.5, np.sin(self.mu), np.cos(self.mu),
               *(u-d),                                   # cap a viser, borne
               *slog(self.pos - self._ref_point(self.t)),  # erreur de suivi
               np.sin(self.turn_ang[i]), np.cos(self.turn_ang[i]),  # virage a venir
               ]
        for j in range(1, LOOKAHEAD+1):
            k = min(i+j, len(self.W)-1)
            v = self.W[k]-self.pos
            obs += list(v/(np.linalg.norm(v)+1e-9))
        rd_, _ = self._ref_state(self.t)
        lag = float((self.pos - self._ref_point(self.t)) @ rd_)   # + = en avance
        # ATTITUDE EXIGEE, maintenant et a l'horizon d'anticipation : ce que
        # l'agent doit COPIER, au lieu de le deviner depuis des positions.
        rd0, rv0 = self._ref_state(self.t)
        rd1, rv1 = self._ref_state(self.t + self.t_lead)
        obs += [*(rd0 - d), *(rd1 - d),
                (rv0 - self.speed)/3.0, (rv1 - self.speed)/3.0]
        obs += [float(slog(dist)), np.clip(t_left, 0, 5)/5.0, v_req,
                float(slog(lag))]
        return np.array(obs, float)

    def _proj(self, i):
        a, b = self.W[i-1], self.W[i]
        ab = b-a
        u = float(np.clip((self.pos-a)@ab/(float(ab@ab)+1e-12), 0.0, 1.0))
        return a + u*ab

    def step(self, action):
        action = np.asarray(action, float)
        om_g, om_c = omega_max(self.speed)
        w, gd, cd, md = weights_from_rates(action[:3], om_g, om_c)
        self.log_w.append(w.copy())
        # vitesse : consigne ABSOLUE dans [V_MIN, V_MAX], glisse a ACC_MAX
        v_cmd = 0.5*(V_MIN+V_MAX) + 0.5*(V_MAX-V_MIN)*np.tanh(action[3])
        n_sub = max(int(round(self.dt_ctrl/self.dt)), 1)
        r_time = 0.0
        for _ in range(n_sub):
            dv = np.clip(v_cmd-self.speed, -ACC_MAX*self.dt, ACC_MAX*self.dt)
            self.speed = float(np.clip(self.speed+dv, V_MIN, V_MAX))
            om_g, om_c = omega_max(self.speed)      # l'enveloppe suit la vitesse
            g2 = np.clip(self.gamma+np.clip(gd,-om_g,om_g)*self.dt,
                         -self.gam_lim, self.gam_lim)
            gd_eff = (g2-self.gamma)/self.dt
            self.gamma = float(g2)
            self.chi += float(np.clip(cd,-om_c,om_c))*self.dt
            self.mu  += float(md)*self.dt
            self.pos = self.pos + self._dir()*self.speed*self.dt
            self.k += 1
            t_before = self.t
            self.t = float(self.T[0]) + self.k*self.dt   # compteur, jamais accumule
            # ponctualite : a l'instant EXACT du beat (le critere du CONTRAT)
            ib = self._target_index(t_before)
            if t_before < self.T[ib] <= self.t:
                e = float(np.linalg.norm(self.pos - self.W[ib]))
                self.beat_err.append(e)
                r_time += W_TIME*np.exp(-e/D_TIME)
            for k_, v_ in [("pos",self.pos.copy()),("t",self.t),("speed",self.speed),
                           ("gamma",self.gamma),("chi",self.chi),("mu",self.mu),
                           ("chi_dot",float(np.clip(cd,-om_c,om_c))),
                           ("gamma_dot",gd_eff),("mu_dot",md)]:
                self.log[k_].append(v_)
        # ---- recompense (3 idees, toutes bornees par pas) ----
        # UN terme dominant : l'erreur au point mobile contient position ET
        # horaire (etre au bon endroit au bon moment EST la meme chose).
        e_ref = float(np.linalg.norm(self.pos - self._ref_point(self.t)))
        r = r_time + W_REF*np.exp(-e_ref/1.5)
        self.track_err.append(e_ref)
        rd_, _ = self._ref_state(self.t)
        lag = abs(float((self.pos - self._ref_point(self.t)) @ rd_))
        r += W_SCHED*(self.prev_lag - lag)          # potentiel (Ng et al.)
        self.prev_lag = lag
        dmu = self.mu - float(self.mu_ref_fn(self.t))
        r += W_MU*np.exp(-abs(np.arctan2(np.sin(dmu), np.cos(dmu))))
        done = self.k >= self.n_steps*max(int(round(self.dt_ctrl/self.dt)),1)
        return self._obs(), float(r), done, {}

    # ------------------------------------------------------------ mesures
    def report(self):
        P = np.asarray(self.log["pos"]); tl = np.asarray(self.log["t"])
        lat = [self._arclen(p)[1] for p in P[::5]]
        dmu = [np.arctan2(np.sin(m-self.mu_ref_fn(t)), np.cos(m-self.mu_ref_fn(t)))
               for m, t in zip(self.log["mu"][::5], tl[::5])]
        # le virage est-il CENTRE sur le beat ? pic de |chi_dot| pres de chaque T_i
        cd = np.abs(np.asarray(self.log["chi_dot"]))
        off = []
        for i in range(1, len(self.T)-1):
            m = (tl > self.T[i]-0.6) & (tl < self.T[i]+0.6)
            if m.sum() > 3 and cd[m].max() > 0.3:
                off.append(abs(float(tl[m][int(cd[m].argmax())]) - self.T[i]))
        return dict(track_err=float(np.mean(self.track_err)) if self.track_err else np.nan,
                    beat_err=float(np.mean(self.beat_err)) if self.beat_err else np.inf,
                    beat_err_max=float(np.max(self.beat_err)) if self.beat_err else np.inf,
                    n_beats=len(self.beat_err),
                    lat_mean=float(np.mean(lat)),
                    mu_err_deg=float(np.degrees(np.mean(np.abs(dmu)))),
                    turn_offset=float(np.mean(off)) if off else np.nan)

    def result(self):
        out = {k: np.asarray(v) for k, v in self.log.items()}
        out["weights"] = np.asarray(self.log_w); out["W"] = self.W; out["T"] = self.T
        return out


# ------------------------------------------------------------------ oracle
def oracle_action(env, k_ang=4.0, k_mu=4.0):
    """Poursuite + loi de vitesse 'arriver a l'heure' : v_des = dist/t_left.

    Borne basse a battre par le RL. Myope : ne ralentit pas avant les coins,
    ne voit pas le virage suivant — exactement la marge laissee au RL.
    """
    i = env._target_index(env.t)
    to = env._ref_point(env.t + env.t_lead) - env.pos   # point MOBILE, devant
    dist = float(np.linalg.norm(to))
    u = to/max(dist, 1e-9)
    gam_des = np.arcsin(np.clip(u[2], -1, 1))
    chi_des = np.arctan2(u[1], u[0])
    e_g = gam_des-env.gamma
    e_c = np.arctan2(np.sin(chi_des-env.chi), np.cos(chi_des-env.chi))
    dmu = env.mu_ref_fn(env.t)-env.mu
    e_m = np.arctan2(np.sin(dmu), np.cos(dmu))
    om_g, om_c = omega_max(env.speed)
    a = np.clip([k_ang*e_g/om_g, k_ang*e_c/om_c, k_mu*e_m/OMEGA_MU], -0.999, 0.999)
    # vitesse de la REFERENCE + rattrapage SIGNE de l'ecart
    if True:
        h = 0.05
        v_ref = float(np.linalg.norm(env._ref_point(env.t+h)-env._ref_point(env.t))/h)
        # correction SIGNEE : lag > 0 = en AVANCE -> ralentir. Avec une norme
        # (toujours positive) l'avion ne pouvait qu'accelerer, prenait de
        # l'avance et devait boucler pour attendre la reference (193 % de
        # longueur volee). L'ecart le long du chemin porte le signe.
        rd, _ = env._ref_state(env.t)
        lag = float((env.pos - env._ref_point(env.t)) @ rd)   # + = en avance
        v_des = np.clip(v_ref - 1.2*lag, V_MIN, V_MAX)
    else:
        t_left = max(float(env.T[i]-env.t), 1e-2)
        v_des = np.clip(dist/t_left, V_MIN, V_MAX)
    a_v = np.arctanh(np.clip((v_des-0.5*(V_MIN+V_MAX))/(0.5*(V_MAX-V_MIN)),
                             -0.999, 0.999))
    return np.concatenate([np.arctanh(a), [a_v]])