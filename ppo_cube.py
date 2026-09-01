"""ETAPE 1 — LE CUBE : le RL bat-il la poursuite pure ?

Cas de test demande par le superviseur : geometrie connue (8 sommets), distances
fixes, instants tres inegaux. On ne cherche pas la beaute, on cherche a savoir si
le SUIVI fonctionne avant de brancher la vraie musique.

CE QU'ON MESURE
---------------
Trois politiques sur le MEME chemin :
    aleatoire  : borne basse, verifie que la tache n'est pas triviale
    oracle     : poursuite pure (viser le prochain waypoint) — la reference
    RL (PPO)   : doit faire MIEUX que l'oracle

Pourquoi le RL devrait gagner : la poursuite est MYOPE. Elle vise le waypoint
courant sans savoir d'ou vient le suivant, donc elle arrive dans chaque virage
mal orientee. Le RL voit LOOKAHEAD=2 waypoints devant : il peut anticiper.
Si le RL ne gagne PAS, le probleme est dans la recompense ou l'observation, et
on le saura sur un cas ou tout le reste est maitrise.

CODE REUTILISE
--------------
Architecture et boucle PPO reprises de brouillon/ppo_blend.py (Gaussienne
diagonale + log_std libre, GAE, clipping) et de archive_rapport/ppo_track.py
(recompense DENSE + decroissance du bonus d'entropie). Rien de neuf ici : seul
l'environnement change.

    python ppo_cube.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn

from test_setup import build
from track_f3p_env import PRIMS
from oracle import pursuit_action


# ----------------------------------------------------------------- reseaux
class ActorCritic(nn.Module):
    """Politique gaussienne + fonction de valeur. Repris tel quel de ppo_blend."""

    def __init__(self, s_dim, a_dim, hidden=64):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(s_dim, hidden), nn.Tanh(),
                                nn.Linear(hidden, hidden), nn.Tanh(),
                                nn.Linear(hidden, a_dim))
        self.v = nn.Sequential(nn.Linear(s_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, 1))
        # log_std : parametre libre, independant de l'etat (pratique standard,
        # plus stable qu'un sigma predit par le reseau)
        self.log_std = nn.Parameter(torch.zeros(a_dim) - 0.5)

    def dist(self, s):
        return torch.distributions.Normal(self.pi(s), self.log_std.exp())

    def act(self, s):
        d = self.dist(s)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.v(s).squeeze(-1)

    def evaluate(self, s, a):
        d = self.dist(s)
        return d.log_prob(a).sum(-1), d.entropy().sum(-1), self.v(s).squeeze(-1)


# ----------------------------------------------------------------- collecte
def collect(env, net, n_ep):
    """Recompense DENSE : un reward a chaque pas, pas seulement a la fin.

    C'est ce qui rend le probleme apprenable ici. Avec un score terminal, le
    credit serait reparti sur 150 pas sans savoir lesquels etaient bons.
    """
    S, A, LP, R, V, DONE = [], [], [], [], [], []
    scores, lat, wp = [], [], []
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
        rep = env.report()
        lat.append(rep["lat_mean"]); wp.append(rep["wp_err_mean"])
    return (np.array(S, np.float32), np.array(A, np.float32), np.array(LP, np.float32),
            np.array(R, np.float32), np.array(V, np.float32), np.array(DONE),
            scores, lat, wp)


def gae(R, V, DONE, gamma=0.97, lam=0.95):
    """Avantage generalise.

    gamma=0.97 et non 1.0 : la recompense est dense et l'episode fait ~150 pas.
    Un gamma de 1 ferait dependre l'avantage d'un pas de tout ce qui suit, y
    compris ce sur quoi ce pas n'a aucune influence — bruit inutile.
    """
    adv = np.zeros_like(R); last = 0.0
    for t in reversed(range(len(R))):
        nextv = 0.0 if DONE[t] else (V[t+1] if t+1 < len(V) else 0.0)
        delta = R[t] + gamma*nextv - V[t]
        last = delta + gamma*lam*(0.0 if DONE[t] else last)
        adv[t] = last
    return adv, adv + V


# ----------------------------------------------------------------- baselines
def run_policy(env, pol):
    env.reset(); done = False; tot = 0.0; n = 0
    while not done:
        _, r, done, _ = env.step(pol())
        tot += r; n += 1
    rep = env.report(); rep["score"] = tot/max(n, 1)
    return rep, env.result()


# ----------------------------------------------------------------- boucle
def train(env, iters=150, ep_per_iter=8, epochs=10, clip=0.2, lr=3e-4,
          ent_coef=0.01, ent_final=5e-4, vf_coef=0.5, seed=0, log_every=25):
    """ent_coef DECROIT geometriquement.

    Avec un bonus constant, sigma reste bloque haut : la politique garde trop de
    bruit pour suivre precisement (observe dans ppo_track.py). On explore au
    debut, on affine ensuite.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    net = ActorCritic(env.state_dim, env.action_dim)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hist = []
    for it in range(iters):
        ec = ent_coef * (ent_final/ent_coef) ** (it/max(iters-1, 1))
        S, A, LP, R, V, DONE, sc, lat, wp = collect(env, net, ep_per_iter)
        adv, ret = gae(R, V, DONE)
        adv = (adv - adv.mean())/(adv.std() + 1e-8)
        S_t, A_t = torch.as_tensor(S), torch.as_tensor(A)
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
        hist.append((np.mean(sc), np.mean(lat), np.mean(wp)))
        if (it+1) % log_every == 0:
            print(f"    iter {it+1:4d} | score {np.mean(sc):.3f} | chemin "
                  f"{np.mean(lat):.2f} m | waypoints {np.mean(wp):.2f} m "
                  f"| sigma {net.log_std.exp().mean():.2f}")
    return net, hist


if __name__ == "__main__":
    ITERS = int(os.environ.get("ITERS", 500))
    EP = int(os.environ.get("EP", 16))
    env, W, T, v_seg, tg, mu = build()
    print(f"cube : {len(W)} waypoints, {T[-1]-T[0]:.1f} s, "
          f"{env.n_steps} pas de controle par episode")
    print(f"state_dim {env.state_dim} | action_dim {env.action_dim}\n")

    rng = np.random.default_rng(0)
    r_rand, _ = run_policy(env, lambda: rng.normal(0, 1.5, env.action_dim))
    r_orc, res_orc = run_policy(env, lambda: pursuit_action(env, k_ang=4.0, k_mu=4.0))

    print("entrainement PPO")
    t0 = time.time()
    net, hist = train(env, iters=ITERS, ep_per_iter=EP, ent_final=1e-5,
                      log_every=max(ITERS//10, 1))
    print(f"  -> {time.time()-t0:.0f} s\n")

    # politique deterministe pour l'evaluation (on prend la moyenne, pas un tirage)
    def greedy():
        with torch.no_grad():
            return net.pi(torch.as_tensor(env._obs(), dtype=torch.float32)).numpy()
    r_rl, res_rl = run_policy(env, greedy)
    W_LOG = np.asarray(env.log_w)          # poids du melange, pour la figure

    # la politique deterministe est-elle meilleure que la politique bruitee ?
    def noisy():
        with torch.no_grad():
            a, _, _ = net.act(torch.as_tensor(env._obs(), dtype=torch.float32))
        return a.numpy()
    lat_n = [run_policy(env, noisy)[0]["lat_mean"] for _ in range(5)]
    print(f"  politique bruitee (5 essais) : {np.mean(lat_n):.2f} m "
          f"+/- {np.std(lat_n):.2f}   |   deterministe : {r_rl['lat_mean']:.2f} m")

    print("="*68)
    print("COMPARAISON  (meme chemin, meme instants)")
    print("="*68)
    print(f"{'politique':<16}{'score':>8}{'chemin':>10}{'waypoints':>12}{'roulis':>10}")
    print("-"*56)
    for name, r in [("aleatoire", r_rand), ("oracle", r_orc), ("RL (PPO)", r_rl)]:
        print(f"{name:<16}{r['score']:>8.3f}{r['lat_mean']:>9.2f} m"
              f"{r['wp_err_mean']:>10.2f} m{r['mu_err_deg']:>8.0f} deg")

    gain = 100*(r_orc["lat_mean"] - r_rl["lat_mean"])/max(r_orc["lat_mean"], 1e-9)
    print(f"\n  ecart au chemin : RL vs oracle -> {gain:+.0f} %")

    np.savez("cube_result.npz", hist=np.array(hist), weights=W_LOG,
             prims=np.array(PRIMS), pos_rl=res_rl["pos"],
             pos_orc=res_orc["pos"], W=W, T=T)
    torch.save(net.state_dict(), "ppo_cube.pt")
    print("  -> cube_result.npz, ppo_cube.pt")