"""ENVIRONNEMENT DE SUIVI — le RL suit un chemin genere par la musique.

SEPARATION DES ROLES
--------------------
  generation (music_path.py) : waypoints + instants + roulis, PUREMENT musical
  suivi (ce fichier)         : le RL oriente l'avion pour suivre, la physique
                               est jugee par le solveur du superviseur

LA VITESSE N'EST PAS UNE ACTION
-------------------------------
Le chemin et les instants etant fixes, v = distance/temps est IMPOSEE. Le RL ne
choisit que la DIRECTION (et le roulis). C'est ensuite l'attitude calculee par le
solveur qui permet de tenir cette vitesse : a basse vitesse l'avion se cabre
(fort angle d'attaque), la poussee le soutient. C'est le regime harrier, prevu
par le modele (ALPHA_BETA_MAX = 89.9 deg).

LES 4 PRIMITIVES (demande du superviseur)
-----------------------------------------
    straight   : rien ne change
    turn       : chi_dot = +/- omega_chi        (droite / gauche)
    pitch      : gamma_dot = +/- omega_gamma    (haut / bas)
    roll       : mu_dot = +/- omega_mu

L'action du RL est un melange pondere de ces taux (softmax), plus les signes.
"""
import numpy as np

# Taux a plein regime. Ordre de grandeur : le rayon de virage vaut v/omega.
# A 6 m/s avec 90 deg/s, le rayon serait de 3.8 m — plus grand que la figure
# elle-meme, l'avion ne pourrait pas prendre un coin. Un F3P de voltige vire
# bien plus vite. Ces taux sont des bornes de COMMANDE ; c'est le solveur du
# superviseur qui juge ensuite si le resultat est volable (residu).
OMEGA_CHI = np.deg2rad(220.0)     # virage
OMEGA_GAM = np.deg2rad(200.0)     # tangage
OMEGA_MU = np.deg2rad(360.0)      # roulis

# les 4 primitives, declinees en + et - (sauf straight)
PRIMS = ["straight",
         "turn_right", "turn_left",
         "pitch_up", "pitch_down",
         "roll_right", "roll_left"]
RATES = {
    "straight":   (0.0,         0.0,        0.0),
    "turn_right": (0.0,        +OMEGA_CHI,  0.0),
    "turn_left":  (0.0,        -OMEGA_CHI,  0.0),
    "pitch_up":   (+OMEGA_GAM,  0.0,        0.0),
    "pitch_down": (-OMEGA_GAM,  0.0,        0.0),
    "roll_right": (0.0,         0.0,       +OMEGA_MU),
    "roll_left":  (0.0,         0.0,       -OMEGA_MU),
}
N_PRIMS = len(PRIMS)
R_MAT = np.array([RATES[p] for p in PRIMS])       # (N_PRIMS, 3)

LOOKAHEAD = 2

# --- modulation de la vitesse par le RL ---
# La vitesse de reference vient de la geometrie (v = distance/temps). Mais la
# suivre STRICTEMENT peut etre infaisable : en vol lent l'avion est en regime
# harrier (la poussee le porte), il ne reste que ~9.8 m/s2 pour accelerer.
# Mesure sur le rectangle : 2 % des instants demandaient plus que la poussee
# disponible, au moment ou il faut accelerer ET virer en meme temps.
#
# On laisse donc le RL AJUSTER la vitesse dans une plage, comme un pilote qui
# arrive legerement en retard plutot que d'exiger l'impossible.
SPEED_MOD = 0.30          # +/- 30 % autour de la reference
ACC_MAX = 5.0             # m/s2 : la vitesse GLISSE vers la consigne, elle ne
                          # saute pas -> acceleration bornee PAR CONSTRUCTION

# --- shaping par potentiel (Ng, Harada & Russell 1999) ---
# MESURE du probleme : le terme de position vaut w_pos*exp(-1.5*e_lat). A 12 m
# du chemin, se rapprocher d'un metre rapporte 5e-8, alors qu'ameliorer le
# roulis rapporte ~1e-1. Le gradient local est un MILLION de fois plus fort sur
# le roulis : l'agent apprend le roulis (6 deg, comme l'oracle) et reste a
# 12.6 m du chemin. Il n'ignore pas la position, il ne la SENT pas.
#
# Correctif : on ajoute F = Phi(s') - Phi(s) avec Phi = -(longueur restante le
# long du chemin). La difference vaut exactement la progression en metres, donc
# le gradient est le MEME a 15 m qu'a 0.5 m.
#
# Le theoreme de Ng et al. garantit que cette forme precise ne change pas la
# politique optimale : on accelere l'apprentissage sans dicter le comportement.
# C'est justement ce qui manquait aux corrections de recompense precedentes.
#
# gamma = 1 dans le shaping (et non 0.97 comme dans GAE) : avec gamma < 1,
# F = k*(rem - gamma*rem') vaut k*0.03*rem meme SANS progresser, ce qui
# recompenserait le simple fait d'etre loin de l'arrivee.
W_PROG = 1.0
# Le potentiel contient AUSSI l'ecart lateral, sinon il reste exploitable :
# l'abscisse curviligne avance meme en volant parallele au chemin a 15 m, donc
# l'agent toucherait toute la progression sans jamais se rapprocher. En mettant
# l'ecart dans le potentiel, s'en rapprocher d'un metre rapporte autant a 15 m
# qu'a 0.5 m.
#
#     Phi(s) = -( longueur restante  +  K_LAT * ecart lateral )
K_LAT = 1.0


def softmax(a):
    a = np.asarray(a, float); a = a - a.max()
    e = np.exp(a); return e / max(e.sum(), 1e-12)


# ------------------------------------------------------- action -> melange
# MESURE (voir MESURES.md M7). Avec w = softmax(logits) et des logits bruites a
# sigma = 0.6, le taux de virage obtenu vaut 7 % du maximum en mediane, alors
# que l'oracle en demande 53 %. Pour atteindre 50 % il faut un ecart de logits
# de 1.9, pour 90 % un ecart de 4.0 : la politique part de logits nuls et le
# BONUS D'ENTROPIE penalise justement les distributions piquees. On lui
# demandait de faire emerger du bruit une structure que son objectif combat.
#
# oracle.py avait deja tranche la question :
#   "Une premiere version notait chaque primitive puis appliquait un softmax.
#    Mauvais [...] Ici on construit les poids directement."
# L'environnement, lui, etait reste au softmax. On aligne les deux.
#
# Le cadrage du superviseur est intact : les 4 primitives existent toujours, les
# poids aussi, "40 % droit / 60 % virage" se lit toujours dans la sortie. Seule
# la PARAMETRISATION change — lineaire au lieu d'exponentielle.

def weights_from_rates(a):
    """a = (a_gam, a_chi, a_mu) dans [-1, 1] -> poids des 7 primitives.

    Chaque axe a une primitive + et une -, et le taux obtenu est proportionnel
    au poids. Si la somme depasse 1, on normalise (l'avion fait de son mieux) ;
    sinon le reste va sur "straight".
    """
    g, c, m = np.tanh(np.asarray(a, float)[:3])
    w = np.zeros(N_PRIMS)
    w[PRIMS.index("pitch_up"   if g > 0 else "pitch_down")] = abs(g)
    w[PRIMS.index("turn_right" if c > 0 else "turn_left")]  = abs(c)
    w[PRIMS.index("roll_right" if m > 0 else "roll_left")]  = abs(m)
    tot = w.sum()
    if tot > 1.0:
        w /= tot
    else:
        w[PRIMS.index("straight")] = 1.0 - tot
    return w


class TrackF3P:
    """Suivi d'un chemin (waypoints + instants + roulis) par melange de primitives."""

    def __init__(self, dt_ctrl=0.1, dt=0.02, gamma_limit_deg=89.0,
                 w_pos=1.0, w_time=1.5, w_mu=0.4, w_smooth=0.05,
                 action_mode="rates"):
        """action_mode : "rates" (defaut) ou "softmax" (ancienne version).

        On garde les deux pour pouvoir les comparer dans le rapport plutot que
        de supprimer une signature qui marchait.
        """
        self.dt_ctrl = dt_ctrl; self.dt = dt
        self.gam_lim = np.deg2rad(gamma_limit_deg)
        self.w_pos, self.w_time = w_pos, w_time
        self.w_mu, self.w_smooth = w_mu, w_smooth
        self.action_mode = action_mode
        self.action_dim = (3 if action_mode == "rates" else N_PRIMS) + 1
        self.state_dim = 6 + 3 + 3 + 3*LOOKAHEAD + 3

    # ------------------------------------------------------------------
    def set_path(self, W, times, mu_ref_fn, v_ref_fn):
        """W : (N,3) waypoints ; times : (N,) instants ; mu_ref_fn, v_ref_fn : t -> valeur."""
        self.W = np.asarray(W, float); self.T = np.asarray(times, float)
        self.mu_ref_fn = mu_ref_fn; self.v_ref_fn = v_ref_fn
        self.n_seg = len(self.W) - 1
        self.seg_len = np.linalg.norm(np.diff(self.W, axis=0), axis=1)
        self.cum_len = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.path_len = float(self.cum_len[-1])
        self.duration = float(self.T[-1] - self.T[0])
        self.n_steps = int(round(self.duration/self.dt_ctrl))

    def _arclen(self, p):
        """Abscisse curviligne du point du CHEMIN le plus proche de p.

        Purement geometrique : elle ne depend pas de l'horloge. C'est ce qui
        rend le potentiel continu au passage d'un waypoint — un potentiel base
        sur le waypoint VISE (qui change avec le temps) sauterait a chaque beat
        et offrirait une recompense gratuite au simple ecoulement du temps.
        """
        best_s, best_d = 0.0, np.inf
        for j in range(self.n_seg):
            a, ab = self.W[j], self.W[j+1] - self.W[j]
            u = float(np.clip((p - a) @ ab / (float(ab @ ab) + 1e-12), 0.0, 1.0))
            d = float(np.linalg.norm(p - (a + u*ab)))
            if d < best_d:
                best_d, best_s = d, float(self.cum_len[j] + u*self.seg_len[j])
        return best_s, best_d

    def _target_index(self, t):
        """Vers quel waypoint on se dirige a l'instant t."""
        return int(np.clip(np.searchsorted(self.T, t, side="right"), 1, len(self.T)-1))

    def _closest_on_path(self, p, i):
        """Point le plus proche sur le segment courant (pour l'erreur laterale)."""
        a, b = self.W[i-1], self.W[i]
        ab = b - a; L2 = float(ab @ ab) + 1e-12
        s = float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
        return a + s*ab, s

    # ------------------------------------------------------------------
    def reset(self):
        self.k = 0                       # numero de sous-pas depuis le debut
        self.t = float(self.T[0])
        self.pos = self.W[0].copy()
        d0 = self.W[1] - self.W[0]; d0 /= np.linalg.norm(d0) + 1e-9
        self.gamma = float(np.arcsin(np.clip(d0[2], -1, 1)))
        self.chi = float(np.arctan2(d0[1], d0[0]))
        self.mu = float(self.mu_ref_fn(self.t))
        self.speed = float(self.v_ref_fn(self.t))
        self.prev_w = np.zeros(N_PRIMS)
        self.prev_s, self.prev_dlat = self._arclen(self.pos)   # potentiel initial
        self.log_w = []                  # poids du melange a chaque pas de controle
        self.log = dict(t=[self.t], pos=[self.pos.copy()],
                        gamma=[self.gamma], chi=[self.chi], mu=[self.mu],
                        speed=[self.speed], gamma_dot=[0.0], chi_dot=[0.0],
                        mu_dot=[0.0], speed_dot=[0.0], weights=[])
        return self._obs()

    def _obs(self):
        i = self._target_index(self.t)
        tgt = self.W[i]
        to_tgt = tgt - self.pos
        dist = np.linalg.norm(to_tgt)
        u = to_tgt/max(dist, 1e-9)
        # direction de vol actuelle
        d = np.array([np.cos(self.gamma)*np.cos(self.chi),
                      np.cos(self.gamma)*np.sin(self.chi),
                      np.sin(self.gamma)])
        proj, _ = self._closest_on_path(self.pos, i)
        lat = self.pos - proj                        # ecart lateral au segment
        t_left = float(self.T[i] - self.t)
        obs = [self.gamma/(np.pi/2), np.sin(self.chi), np.cos(self.chi),
               (self.speed - 3.0)/3.0, np.sin(self.mu), np.cos(self.mu),
               *(u - d),                              # ecart de cap a viser (3)
               *(lat/2.0),                            # ecart lateral (3)
               ]
        for j in range(1, LOOKAHEAD+1):
            k = min(i+j, len(self.W)-1)
            v = self.W[k] - self.pos
            obs += list(v/ (np.linalg.norm(v)+1e-9))
        mu_err = np.arctan2(np.sin(self.mu - self.mu_ref_fn(self.t)),
                            np.cos(self.mu - self.mu_ref_fn(self.t)))
        v_ref = float(self.v_ref_fn(self.t))
        obs += [dist/5.0, np.clip(t_left, 0, 5)/5.0,
                (self.speed - v_ref)/2.0]     # ecart a la vitesse de reference
        return np.array(obs, float)

    # ------------------------------------------------------------------
    def step(self, action):
        # cible AVANT d'avancer : apres le pas, _target_index pointe deja sur le
        # waypoint suivant, donc le test de franchissement ne se declencherait
        # jamais si on le calculait apres.
        i_before = self._target_index(self.t)
        t_before = self.t
        action = np.asarray(action, float)
        if self.action_mode == "rates":
            w = weights_from_rates(action[:3])
        else:
            w = softmax(action[:N_PRIMS])
        self.log_w.append(w.copy())
        mod = 1.0 + SPEED_MOD*np.tanh(action[-1])       # consigne de vitesse
        rates = w @ R_MAT                            # (gamma_dot, chi_dot, mu_dot)
        n_sub = max(int(round(self.dt_ctrl/self.dt)), 1)
        for _ in range(n_sub):
            g_new = float(np.clip(self.gamma + rates[0]*self.dt,
                                  -self.gam_lim, self.gam_lim))
            gd = (g_new - self.gamma)/self.dt        # taux EFFECTIF (borne prise en compte)
            self.gamma = g_new
            self.chi += rates[1]*self.dt
            self.mu += rates[2]*self.dt
            v_cmd = float(self.v_ref_fn(self.t + self.dt))*mod
            # la vitesse GLISSE vers la consigne : acceleration bornee
            dv = float(np.clip(v_cmd - self.speed, -ACC_MAX*self.dt, ACC_MAX*self.dt))
            v_new = max(self.speed + dv, 0.0)
            sd = (v_new - self.speed)/self.dt
            self.speed = v_new
            d = np.array([np.cos(self.gamma)*np.cos(self.chi),
                          np.cos(self.gamma)*np.sin(self.chi),
                          np.sin(self.gamma)])
            self.pos = self.pos + d*self.speed*self.dt
            # t CALCULE, pas accumule. Avec self.t += self.dt, 750 additions de
            # 0.02 (non representable en binaire) laissent 2.3e-13 s d'erreur :
            # le test de franchissement T[i] <= t echouait au DERNIER waypoint,
            # qui ne recevait donc jamais sa recompense de synchronisation.
            self.k += 1
            self.t = float(self.T[0]) + self.k*self.dt
            for k, v in [("t",self.t), ("pos",self.pos.copy()), ("gamma",self.gamma),
                         ("chi",self.chi), ("mu",self.mu), ("speed",self.speed),
                         ("gamma_dot",gd), ("chi_dot",rates[1]),
                         ("mu_dot",rates[2]), ("speed_dot",sd)]:
                self.log[k].append(v)
        self.log["weights"].append(w)

        # ---------------- recompense ----------------
        i = self._target_index(self.t)
        proj, _ = self._closest_on_path(self.pos, i)
        e_lat = float(np.linalg.norm(self.pos - proj))          # ecart au chemin
        r = self.w_pos*np.exp(-1.5*e_lat)

        # progression le long du chemin : gradient identique a toute distance
        s_now, dlat_now = self._arclen(self.pos)
        r += W_PROG*(s_now - self.prev_s) + K_LAT*(self.prev_dlat - dlat_now)
        self.prev_s, self.prev_dlat = s_now, dlat_now

        # passage au waypoint a l'instant du beat
        r_time = 0.0
        if t_before < self.T[i_before] <= self.t:      # on vient de franchir l'instant
            # position la plus proche du waypoint pendant ce pas de controle
            P = np.asarray(self.log["pos"][-(int(round(self.dt_ctrl/self.dt))+1):])
            e_wp = float(np.linalg.norm(P - self.W[i_before], axis=1).min())
            r_time = self.w_time*np.exp(-1.0*e_wp)
            self.log.setdefault("wp_err", []).append(e_wp)
        r += r_time

        # suivi du roulis musical
        dmu = self.mu - float(self.mu_ref_fn(self.t))
        e_mu = abs(np.arctan2(np.sin(dmu), np.cos(dmu)))
        r += self.w_mu*np.exp(-1.5*e_mu)

        # rester proche de la vitesse de reference (sinon le RL ralentirait
        # systematiquement : plus facile a suivre, mais on arrive en retard)
        v_ref = float(self.v_ref_fn(self.t))
        r += 0.3*np.exp(-2.0*abs(self.speed - v_ref)/max(v_ref, 0.5))

        # douceur : eviter que le melange saute d'un pas a l'autre
        r -= self.w_smooth*float(np.abs(w - self.prev_w).sum())
        self.prev_w = w

        done = self.t >= self.T[-1] - 1e-9
        return self._obs(), float(r), done, dict(e_lat=e_lat)

    # ------------------------------------------------------------------
    def result(self):
        out = {k: np.asarray(v) for k, v in self.log.items()
               if k not in ("weights", "wp_err")}
        out["weights"] = np.asarray(self.log["weights"])
        out["wp_err"] = np.asarray(self.log.get("wp_err", []))
        return out

    def report(self):
        r = self.result()
        P = r["pos"]
        lat = []
        for j, t in enumerate(r["t"]):
            i = self._target_index(float(t))
            proj, _ = self._closest_on_path(P[j], i)
            lat.append(np.linalg.norm(P[j]-proj))
        mu_e = []
        for j, t in enumerate(r["t"]):
            d = r["mu"][j] - float(self.mu_ref_fn(float(t)))
            mu_e.append(abs(np.arctan2(np.sin(d), np.cos(d))))
        return dict(lat_mean=float(np.mean(lat)), lat_max=float(np.max(lat)),
                    wp_err_mean=float(np.mean(r["wp_err"])) if len(r["wp_err"]) else np.nan,
                    mu_err_deg=float(np.degrees(np.mean(mu_e))),
                    speed_min=float(r["speed"].min()), speed_max=float(r["speed"].max()))