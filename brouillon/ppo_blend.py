"""PPO CONTINU pour la composition de figures F3P par melange.

POURQUOI PPO CONTINU (et pas Categorical)
-----------------------------------------
L'action n'est plus "choisir une primitive dans un catalogue" (discret) mais
"doser le melange" : un vecteur reel de dimension N_PRIMS, transforme en poids
par softmax. La distribution adaptee est donc une GAUSSIENNE diagonale :

    a ~ N( mu_theta(s), diag(sigma_theta) )        puis   w = softmax(a)

La politique apprend la moyenne mu_theta(s) et l'ecart-type sigma_theta
(independant de l'etat, parametre par log_std : c'est la pratique standard, plus
stable qu'un sigma dependant de l'etat).

L'OBJECTIF PPO
--------------
    L(theta) = E[ min( r_t(theta) A_t , clip(r_t(theta), 1-eps, 1+eps) A_t ) ]

    r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)

Le clipping empeche la politique de trop s'eloigner de l'ancienne a chaque mise
a jour : c'est ce qui rend PPO stable la ou REINFORCE oscille.

On y ajoute :
  - une fonction de VALEUR V_phi(s) pour l'avantage (GAE)
  - un bonus d'ENTROPIE pour maintenir l'exploration

    python ppo_blend.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn

from blend_env import BlendEnv
from blend import PRIMS, N_PRIMS

# ----------------------------------------------------------------- solveur
try:
    from f3p_attitude.solver import solve_trajectory
    from f3p_attitude.constants import THRUST_MAX
    def solver(seq):
        try:
            r = solve_trajectory(seq["t"], seq["gamma"], seq["chi"], seq["speed"],
                                 seq["mu"], gamma_dot=seq["gamma_dot"],
                                 chi_dot=seq["chi_dot"], speed_dot=seq["speed_dot"])
        except Exception:
            return None
        return (float(np.asarray(r.thrust).max()/THRUST_MAX),
                float(np.degrees(np.abs(np.asarray(r.alpha))).max()),
                float(np.asarray(r.residual_norm).max()))
    SOLVER_NAME = "f3p_attitude (physique reelle)"
except ImportError:
    from mock_solver import mock as solver
    SOLVER_NAME = "SIMULE (f3p_attitude introuvable)"


# ----------------------------------------------------------------- reseaux
class ActorCritic(nn.Module):
    def __init__(self, s_dim, a_dim, hidden=64):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(s_dim, hidden), nn.Tanh(),
                                nn.Linear(hidden, hidden), nn.Tanh(),
                                nn.Linear(hidden, a_dim))
        self.v = nn.Sequential(nn.Linear(s_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, 1))
        # log_std : parametre libre, independant de l'etat (pratique standard)
        self.log_std = nn.Parameter(torch.zeros(a_dim) - 0.5)

    def dist(self, s):
        mu = self.pi(s)
        return torch.distributions.Normal(mu, self.log_std.exp())

    def act(self, s):
        d = self.dist(s)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.v(s).squeeze(-1)

    def evaluate(self, s, a):
        d = self.dist(s)
        return (d.log_prob(a).sum(-1), d.entropy().sum(-1), self.v(s).squeeze(-1))


# ----------------------------------------------------------------- collecte
def collect(env, net, n_episodes, device):
    S, A, LP, R, V, DONE = [], [], [], [], [], []
    rewards, feas = [], []
    for _ in range(n_episodes):
        s = env.reset(); done = False
        ep_s, ep_a, ep_lp, ep_v = [], [], [], []
        while not done:
            st = torch.as_tensor(s, dtype=torch.float32, device=device)
            with torch.no_grad():
                a, lp, v = net.act(st)
            ep_s.append(s); ep_a.append(a.cpu().numpy())
            ep_lp.append(float(lp)); ep_v.append(float(v))
            s, done = env.step(a.cpu().numpy())
        Rtot, info = env.evaluate(env.build(), solver)
        rewards.append(Rtot); feas.append(1.0 if info["feasible"] else 0.0)
        # recompense terminale : tous les pas de l'episode partagent le retour
        n = len(ep_s)
        S += ep_s; A += ep_a; LP += ep_lp; V += ep_v
        R += [0.0]*(n-1) + [Rtot]
        DONE += [False]*(n-1) + [True]
    return (np.array(S, np.float32), np.array(A, np.float32),
            np.array(LP, np.float32), np.array(R, np.float32),
            np.array(V, np.float32), np.array(DONE), rewards, feas)


def gae(R, V, DONE, gamma=1.0, lam=0.95):
    """Avantage generalise. gamma=1 : on optimise le retour total de la figure."""
    adv = np.zeros_like(R); last = 0.0
    for t in reversed(range(len(R))):
        nextv = 0.0 if DONE[t] else V[t+1]
        delta = R[t] + gamma*nextv - V[t]
        last = delta + gamma*lam*(0.0 if DONE[t] else last)
        adv[t] = last
    return adv, adv + V


# ----------------------------------------------------------------- boucle
def train(iters=60, episodes_per_iter=24, epochs=8, clip=0.2, lr=3e-4,
          ent_coef=0.01, vf_coef=0.5, seed=0, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)
    env = BlendEnv(n_control=5, duration=5.0)
    net = ActorCritic(env.state_dim, env.action_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hist = []
    for it in range(iters):
        S, A, LP, R, V, DONE, rew, feas = collect(env, net, episodes_per_iter, device)
        adv, ret = gae(R, V, DONE)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        S_t = torch.as_tensor(S, device=device); A_t = torch.as_tensor(A, device=device)
        LP_t = torch.as_tensor(LP, device=device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=device)
        for _ in range(epochs):
            lp, ent, v = net.evaluate(S_t, A_t)
            ratio = (lp - LP_t).exp()
            l1 = ratio * adv_t
            l2 = torch.clamp(ratio, 1-clip, 1+clip) * adv_t
            loss = -(torch.min(l1, l2).mean()) \
                   + vf_coef*((v - ret_t)**2).mean() \
                   - ent_coef*ent.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
        hist.append((np.mean(rew), np.mean(feas)))
        if (it+1) % 5 == 0:
            print(f"    iter {it+1:3d} | reward {np.mean(rew):+.3f} "
                  f"| volables {100*np.mean(feas):3.0f}% "
                  f"| sigma {net.log_std.exp().mean().item():.2f}")
    return net, env, hist


if __name__ == "__main__":
    print(f"solveur : {SOLVER_NAME}")
    print(f"action continue de dimension {N_PRIMS} -> softmax -> poids du melange\n")
    t0 = time.time()
    net, env, hist = train()
    print(f"\nentraine en {time.time()-t0:.0f}s")
    print(f"  reward   : {hist[0][0]:+.3f} -> {hist[-1][0]:+.3f}")
    print(f"  volables : {100*hist[0][1]:.0f}% -> {100*hist[-1][1]:.0f}%")
    torch.save(net.state_dict(), "ppo_blend.pt")

    print("\n--- meilleures figures ---")
    best = []
    for _ in range(40):
        s = env.reset(); done = False
        while not done:
            with torch.no_grad():
                a = net.pi(torch.as_tensor(s, dtype=torch.float32))
            s, done = env.step(a.numpy())
        seq = env.build(); R, info = env.evaluate(seq, solver)
        if info["feasible"]:
            best.append((R, seq, info))
    best.sort(key=lambda x: -x[0])
    for rank, (R, seq, info) in enumerate(best[:3], 1):
        print(f"\n  #{rank} reward {R:+.3f} | poussee {100*info['thrust']:.0f}% "
              f"| diversite {info['s_div']:.2f} | variation {info['s_var']:.2f}")
        print(f"      melange moyen : {info['wbar']}")
    if best:
        np.save("best_weights.npy", best[0][1]["weights"])
        print("\n  poids de la meilleure figure -> best_weights.npy")