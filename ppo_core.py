"""PPO unique du projet (ppo_core) — importe partout, copie nulle part.

Architecture et hyperparametres repris a l'identique de la boucle qui a fait ses
preuves (ppo_blend -> ppo_track -> ppo_cube) : gaussienne diagonale a log_std
libre, GAE(0.97, 0.95), clipping 0.2, bonus d'entropie DECROISSANT (sinon sigma
reste bloque haut et la politique garde trop de bruit pour suivre finement).

L'environnement doit fournir : reset() -> obs, step(a) -> (obs, r, done, info),
state_dim, action_dim, et idealement report() -> dict de metriques.
"""
import numpy as np
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:                     # ResidualWrapper reste utilisable
    HAS_TORCH = False


class ActorCritic(nn.Module if HAS_TORCH else object):
    def __init__(self, s_dim, a_dim, hidden=64):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(s_dim, hidden), nn.Tanh(),
                                nn.Linear(hidden, hidden), nn.Tanh(),
                                nn.Linear(hidden, a_dim))
        self.v = nn.Sequential(nn.Linear(s_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, 1))
        # -1.4 (sigma~0.25) et non -0.5 : en residuel l'exploration est LOCALE
        # autour d'un controleur deja competent, un gros bruit detruirait la
        # base au lieu de l'ameliorer.
        self.log_std = nn.Parameter(torch.zeros(a_dim) - 1.4)

    def dist(self, s):
        return torch.distributions.Normal(self.pi(s), self.log_std.exp())

    def act(self, s):
        d = self.dist(s); a = d.sample()
        return a, d.log_prob(a).sum(-1), self.v(s).squeeze(-1)

    def evaluate(self, s, a):
        d = self.dist(s)
        return d.log_prob(a).sum(-1), d.entropy().sum(-1), self.v(s).squeeze(-1)


class ResidualWrapper:
    """RL RESIDUEL (Silver et al. ; Johannink et al.) : a = base(s) + delta.

    Le motif des runs from scratch : la politique bruitee meilleure que la
    moyenne, sigma bloque, degradation en fin d'entrainement, greedy qui
    s'effondre sur un beat. Cause : l'oracle est une fonction LINEAIRE de
    grandeurs presentes dans l'observation — on demandait a PPO de redecouvrir
    par essai-erreur une formule qu'on possede.

    Ici le reseau n'apprend que la CORRECTION delta autour du controleur de
    base : a l'initialisation (delta~0) la politique EST l'oracle, et
    l'exploration est locale autour d'un pilote competent. Le RL apprend ce qui
    lui revient : ce que la poursuite myope ne peut pas faire.
    """

    def __init__(self, env, base_policy, delta_scale=1.0):
        self.env, self.base, self.scale = env, base_policy, delta_scale
        self.state_dim, self.action_dim = env.state_dim, env.action_dim

    def reset(self):
        return self.env.reset()

    def step(self, delta):
        a = self.base(self.env) + self.scale*np.asarray(delta, float)
        return self.env.step(a)

    def _obs(self):
        return self.env._obs()

    def report(self):
        return self.env.report()

    def result(self):
        return self.env.result()


def collect(env, net, n_ep, metric_keys=()):
    S, A, LP, R, V, DONE = [], [], [], [], [], []
    scores = []; mets = {k: [] for k in metric_keys}
    for _ in range(n_ep):
        s = env.reset(); done = False; tot = 0.0; n = 0
        while not done:
            st = torch.as_tensor(s, dtype=torch.float32)
            with torch.no_grad():
                a, lp, v = net.act(st)
            S.append(s); A.append(a.numpy()); LP.append(float(lp)); V.append(float(v))
            s, r, done, _ = env.step(a.numpy())
            R.append(r); DONE.append(done); tot += r; n += 1
        scores.append(tot/max(n, 1))
        if metric_keys:
            rep = env.report()
            for k in metric_keys: mets[k].append(rep[k])
    return (np.array(S, np.float32), np.array(A, np.float32),
            np.array(LP, np.float32), np.array(R, np.float32),
            np.array(V, np.float32), np.array(DONE), scores, mets)


def gae(R, V, DONE, gamma=0.97, lam=0.95):
    adv = np.zeros_like(R); last = 0.0
    for t in reversed(range(len(R))):
        nextv = 0.0 if DONE[t] else (V[t+1] if t+1 < len(V) else 0.0)
        delta = R[t] + gamma*nextv - V[t]
        last = delta + gamma*lam*(0.0 if DONE[t] else last)
        adv[t] = last
    return adv, adv + V


def train(env, iters=400, ep_per_iter=16, epochs=10, clip=0.2, lr=3e-4,
          ent_coef=0.01, ent_final=1e-5, vf_coef=0.5, seed=0,
          metric_keys=(), log_every=50):
    torch.manual_seed(seed); np.random.seed(seed)
    net = ActorCritic(env.state_dim, env.action_dim)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hist = []
    for it in range(iters):
        ec = ent_coef * (ent_final/ent_coef) ** (it/max(iters-1, 1))
        S, A, LP, R, V, DONE, sc, mets = collect(env, net, ep_per_iter, metric_keys)
        adv, ret = gae(R, V, DONE)
        adv = (adv - adv.mean())/(adv.std() + 1e-8)
        S_t = torch.as_tensor(S); A_t = torch.as_tensor(A)
        LP_t = torch.as_tensor(LP)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)
        for _ in range(epochs):
            lp, ent, v = net.evaluate(S_t, A_t)
            ratio = (lp - LP_t).exp()
            loss = -torch.min(ratio*adv_t,
                              torch.clamp(ratio, 1-clip, 1+clip)*adv_t).mean() \
                   + vf_coef*((v-ret_t)**2).mean() - ec*ent.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        row = [np.mean(sc)] + [float(np.mean(mets[k])) for k in metric_keys]
        hist.append(row)
        if (it+1) % log_every == 0:
            extra = " | ".join(f"{k} {float(np.mean(mets[k])):.2f}" for k in metric_keys)
            print(f"    iter {it+1:4d} | score {np.mean(sc):.3f} | {extra} "
                  f"| sigma {net.log_std.exp().mean():.2f}")
    return net, np.array(hist)


def greedy_policy(env, net):
    def pol():
        with torch.no_grad():
            return net.pi(torch.as_tensor(env._obs(), dtype=torch.float32)).numpy()
    return pol


def run_policy(env, pol):
    env.reset(); done = False; tot = 0.0; n = 0
    while not done:
        _, r, done, _ = env.step(pol()); tot += r; n += 1
    rep = env.report(); rep["score"] = tot/max(n, 1)
    return rep, env.result()