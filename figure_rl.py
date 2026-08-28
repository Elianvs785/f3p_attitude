"""RL de COMPOSITION DE FIGURES F3P.

CE QU'IL APPREND
----------------
A enchainer des primitives (droit, virage, montee, tranche, tonneau) pour former
une figure qui soit :
  - VOLABLE  : l'equilibre des forces est satisfait (residu faible), pas de
               decrochage, vitesse de roulis plausible
  - SPECTACULAIRE : qui exploite la marge de poussee sans la depasser, et qui
               varie les primitives au lieu de repeter la meme.

L'ESPACE D'ACTION
-----------------
Un CATALOGUE de mouvements, chacun avec des parametres pris dans les plages
mesurees par explore_limits.py :
    vitesse    3 - 7 m/s      (<= 5 si virage serre)
    virage     >= 0.5 s pour 90 deg
    vertical   >= 0.7 s pour 90 deg
    tonneau    <= 720 deg/s
    tranche    3 - 7 m/s
Le RL ne peut donc proposer que du realisable : il apprend a COMBINER, pas a
redecouvrir la physique.

LA RECOMPENSE
-------------
    figure involable            -> -1
    sinon  0.5 * score_poussee            (marge exploitee, cible ~85%)
         + 0.3 * score_diversite          (varier les primitives)
         + 0.2 * score_amplitude          (angles parcourus)

Le score de poussee est le coeur : une figure a 30% de poussee est molle, une a
100% ne vole pas. L'interet est dans la zone 70-95%.
"""
import numpy as np
from compose import initial_state, sequence, to_solver_input

# ------------------------------------------------------------------ catalogue
# (nom_primitive, kwargs, vitesse_cible)  — bornes issues de explore_limits.py
CATALOG = [
    ("straight",   {},                                    5.0),
    ("straight",   {"speed": 4.0},                        4.0),
    ("straight",   {"speed": 6.5},                        6.5),
    ("turn",       {"delta_chi_deg":  90.0},              5.0),
    ("turn",       {"delta_chi_deg": -90.0},              5.0),
    ("turn",       {"delta_chi_deg": 180.0},              4.5),
    ("turn",       {"delta_chi_deg":  45.0},              5.5),
    ("climb",      {"delta_gamma_deg":  90.0},            5.0),
    ("climb",      {"delta_gamma_deg": -90.0},            5.0),
    ("climb",      {"delta_gamma_deg":  45.0},            5.5),
    ("climb",      {"delta_gamma_deg": -45.0},            5.5),
    ("knife_edge", {"mu_deg":  90.0},                     5.0),
    ("knife_edge", {"mu_deg": -90.0},                     5.0),
    ("roll",       {"n_turns": 1.0},                      5.0),
    ("roll",       {"n_turns": -1.0},                     5.0),
]
# durees admissibles par primitive (secondes) — marges de explore_limits
DURATIONS = {"straight": [0.4, 0.8, 1.2],
             "turn":     [0.6, 0.9, 1.4],
             "climb":    [0.8, 1.2, 1.8],
             "knife_edge": [1.2, 2.0],
             "roll":     [1.2, 1.8]}

MOVES = [(i, d) for i, (name, kw, sp) in enumerate(CATALOG)
         for d in DURATIONS[name]]
N_MOVES = len(MOVES)
TYPES = ["straight", "turn", "climb", "knife_edge", "roll"]


def move_to_step(move):
    """Un mouvement du catalogue -> une etape pour sequence()."""
    i, dur = move
    name, kw, speed = CATALOG[i]
    kw = dict(kw)
    if name in ("straight",) and "speed" not in kw:
        pass
    elif name in ("turn", "climb"):
        kw["speed"] = speed          # ralentir/accelerer pendant la figure
    return (name, dur, kw)


# --------------------------------------------------------------- evaluation
def build(moves, speed0=5.0, dt=0.05):
    st = initial_state(speed=speed0)
    return sequence(st, [move_to_step(m) for m in moves], dt=dt)


def integrate_position(seq):
    """Reconstruit la trajectoire (z vers le haut), comme integrate_position
    de scenarios.py."""
    t = seq["t"]; g = seq["gamma"]; c = seq["chi"]; s = seq["speed"]
    v = np.stack([s*np.cos(g)*np.cos(c), s*np.cos(g)*np.sin(c), s*np.sin(g)], axis=1)
    p = np.zeros_like(v)
    for i in range(len(t)-1):
        p[i+1] = p[i] + 0.5*(v[i]+v[i+1])*(t[i+1]-t[i])
    return p


def score_figure(seq, solver, residual_tol=1e-4, alpha_stall=25.0,
                 roll_max_deg=720.0, thrust_target=0.85,
                 box=(40.0, 40.0, 10.0)):
    """Evalue une figure : volable ? spectaculaire ? Retourne (reward, infos)."""
    # --- CONTRAINTE DE VOLUME ---
    # Le F3P se vole en salle. Sans cette contrainte, la politique produisait des
    # piques verticaux descendant a -28 m : la recompense etait satisfaite
    # (poussee 87 %) mais la figure etait absurde.
    #
    # Calibration mesuree du taux de figures volables en fin d'entrainement :
    #     20x20x8   ->  2 %      30x30x10 ->  6 %      40x40x10 -> 66 %
    # En dessous de ~40 m, trop peu d'exemples positifs pour que l'apprentissage
    # demarre (recompense trop eparse) : les rares figures qui tiennent sont
    # excellentes mais introuvables. 40x40x10 est le compromis retenu ; resserrer
    # `box` demanderait des figures plus courtes ou plus lentes.
    P = integrate_position(seq)
    P[:, 2] -= P[:, 2].min()                     # depart au sol
    span = P.max(axis=0) - P.min(axis=0)
    if span[0] > box[0] or span[1] > box[1] or span[2] > box[2]:
        return -1.0, dict(feasible=False, why="hors salle", thrust=np.nan,
                          alpha=np.nan, res=np.nan, roll=np.nan,
                          span=tuple(np.round(span, 1)))

    roll_rate = float(np.degrees(np.abs(seq["mu_dot"]).max()))
    if roll_rate > roll_max_deg:
        return -1.0, dict(feasible=False, why="roulis", thrust=np.nan,
                          alpha=np.nan, res=np.nan, roll=roll_rate)
    r = solver(seq)
    if r is None:
        return -1.0, dict(feasible=False, why="solveur", thrust=np.nan,
                          alpha=np.nan, res=np.nan, roll=roll_rate)
    thrust, alpha, res = r
    if res > residual_tol:
        return -1.0, dict(feasible=False, why="equilibre", thrust=thrust,
                          alpha=alpha, res=res, roll=roll_rate)
    if alpha > alpha_stall:
        return -1.0, dict(feasible=False, why="decrochage", thrust=thrust,
                          alpha=alpha, res=res, roll=roll_rate)

    # --- figure volable : on note le "spectacle" ---
    # Lecon apprise : sans garde-fou, la politique repete la primitive la moins
    # couteuse (6 tonneaux d'affilee) car elle saturait le score d'amplitude.
    # On renforce donc la diversite et on penalise la repetition.
    s_thrust = 1.0 - abs(thrust - thrust_target) / thrust_target
    s_thrust = float(np.clip(s_thrust, 0.0, 1.0))

    names = [CATALOG[i][0] for i, _ in seq.get("moves", [])]
    n = len(names)

    # --- diversite : combien de TYPES differents sur les 5 possibles ---
    s_div = len(set(names)) / len(TYPES) if names else 0.0

    # --- non-repetition : deux mesures complementaires ---
    # (a) repetitions CONSECUTIVES (turn, turn, turn...)
    rep_consec = sum(1 for a, b in zip(names, names[1:]) if a == b)
    s_consec = 1.0 - rep_consec / max(n - 1, 1)
    # (b) DESEQUILIBRE global : 4 virages separes par une ligne droite echappaient
    #     a la mesure (a) alors que le resultat est tout aussi monotone.
    #     On mesure a quel point une primitive domine la figure.
    if n:
        counts = np.array([names.count(t) for t in TYPES], float)
        share_max = counts.max() / n                 # 1/n = parfaitement reparti
        s_balance = float(np.clip((1.0 - share_max) / (1.0 - 1.0/len(TYPES)), 0, 1))
    else:
        s_balance = 0.0
    s_rep = 0.5 * s_consec + 0.5 * s_balance

    # amplitude : chaque axe normalise SEPAREMENT, puis moyenne
    a_chi = np.clip(np.abs(np.diff(seq["chi"])).sum() / (2*np.pi), 0, 1)
    a_gam = np.clip(np.abs(np.diff(seq["gamma"])).sum() / np.pi, 0, 1)
    a_mu = np.clip(np.abs(np.diff(seq["mu"])).sum() / (2*np.pi), 0, 1)
    s_amp = float((a_chi + a_gam + a_mu) / 3.0)

    reward = 0.35*s_thrust + 0.25*s_div + 0.30*s_rep + 0.10*s_amp
    return float(reward), dict(feasible=True, why="", thrust=thrust, alpha=alpha,
                               res=res, roll=roll_rate, s_thrust=s_thrust,
                               s_div=s_div, s_rep=s_rep, s_amp=s_amp,
                               s_consec=s_consec, s_balance=s_balance,
                               span=tuple(np.round(span, 1)))


# ------------------------------------------------------------------ politique
F_DIM = 8


def features(state, step, n_steps, used):
    """Etat de vol + avancement + primitives deja utilisees."""
    return np.array([
        state["gamma"] / (np.pi/2), np.sin(state["chi"]), np.cos(state["chi"]),
        (state["speed"] - 5.0) / 2.0, state["mu"] / np.pi,
        step / max(n_steps, 1),
        len(used) / len(TYPES),
        1.0,
    ], float)


class Policy:
    def __init__(self, hidden=32, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.4, (F_DIM, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.4, (hidden, N_MOVES)); self.b2 = np.zeros(N_MOVES)

    def forward(self, f):
        h = np.tanh(f @ self.W1 + self.b1)
        s = h @ self.W2 + self.b2
        s = s - s.max()
        e = np.exp(s)
        return e / e.sum(), h

    def grads(self, f, h, p, action):
        d = -p.copy(); d[action] += 1.0
        gW2 = np.outer(h, d); gb2 = d
        dh = (self.W2 @ d) * (1 - h**2)
        return np.outer(f, dh), dh, gW2, gb2

    def apply(self, g, lr):
        self.W1 += lr*g[0]; self.b1 += lr*g[1]; self.W2 += lr*g[2]; self.b2 += lr*g[3]


# ------------------------------------------------------------- entrainement
def rollout(pol, rng, n_steps, solver, speed0=5.0, greedy=False):
    """Construit une figure en choisissant n_steps mouvements, puis l'evalue."""
    st = initial_state(speed=speed0)
    traj = []; moves = []; used = set()
    for k in range(n_steps):
        f = features(st, k, n_steps, used)
        p, h = pol.forward(f)
        a = int(np.argmax(p)) if greedy else int(rng.choice(N_MOVES, p=p))
        traj.append((f, h, p, a))
        mv = MOVES[a]; moves.append(mv)
        used.add(CATALOG[mv[0]][0])
        # avancer l'etat de vol
        seq_k = sequence(st, [move_to_step(mv)], dt=0.05)
        st = dict(gamma=float(seq_k["gamma"][-1]), chi=float(seq_k["chi"][-1]),
                  speed=float(seq_k["speed"][-1]), mu=float(seq_k["mu"][-1]))
    seq = build(moves, speed0=speed0)
    seq["moves"] = moves
    reward, info = score_figure(seq, solver)
    return reward, traj, moves, seq, info


def train(solver, episodes=400, n_steps=6, lr=0.05, seed=0, log_every=50):
    pol = Policy(seed=seed); rng = np.random.default_rng(seed)
    baseline = None; hist = []; feas = []
    for ep in range(episodes):
        R, traj, moves, seq, info = rollout(pol, rng, n_steps, solver)
        hist.append(R); feas.append(1.0 if info["feasible"] else 0.0)
        baseline = R if baseline is None else 0.9*baseline + 0.1*R
        adv = R - baseline
        g = [np.zeros_like(pol.W1), np.zeros_like(pol.b1),
             np.zeros_like(pol.W2), np.zeros_like(pol.b2)]
        for f, h, p, a in traj:
            gi = pol.grads(f, h, p, a)
            for k in range(4): g[k] += adv * gi[k]
        pol.apply(g, lr/len(traj))
        if log_every and (ep+1) % log_every == 0:
            print(f"    episode {ep+1:4d} | reward {np.mean(hist[-50:]):+.3f} "
                  f"| volables {100*np.mean(feas[-50:]):3.0f}%")
    return pol, hist, feas