"""PPO CONTINU pour le SUIVI DE TRAJECTOIRE par melange de primitives.

On donne une trajectoire cible ; le RL doit trouver, a chaque instant, le
melange de primitives qui la reproduit.

DIFFERENCE AVEC ppo_blend.py (inventer une figure)
--------------------------------------------------
  recompense DENSE (erreur de suivi a chaque pas) au lieu d'un score terminal
  -> le credit est bien mieux attribue
  le melange DOIT varier dans le temps -> resout le "variation = 0.00" observe

    python ppo_track.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn

from track_env import TrackEnv, reference_from_weights
from blend import PRIMS, N_PRIMS


# ---------------------------------------------- generateur de references
from references import FIGURES, make, make_random

def random_reference(rng, dt=0.02):
    """Reference d'entrainement : moitie figures de la bibliotheque, moitie
    aleatoires STRUCTUREES.

    Les melanges Dirichlet diffus (alpha=1) donnaient des trajectoires quasi
    rectilignes : les primitives se compensaient. On tire donc des melanges
    tranches (alpha=0.35) et on melange avec de vraies figures de voltige, pour
    que la politique apprenne a suivre des manoeuvres reelles.
    """
    if rng.random() < 0.5:
        return make(list(FIGURES)[rng.integers(len(FIGURES))], dt=dt)
    return make_random(rng, dt=dt, duration=float(rng.uniform(6.0, 10.0)))


class ActorCritic(nn.Module):
    def __init__(self, s_dim, a_dim, hidden=96):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(s_dim, hidden), nn.Tanh(),
                                nn.Linear(hidden, hidden), nn.Tanh(),
                                nn.Linear(hidden, a_dim))
        self.v = nn.Sequential(nn.Linear(s_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.zeros(a_dim) - 0.5)

    def dist(self, s):
        return torch.distributions.Normal(self.pi(s), self.log_std.exp())

    def act(self, s):
        d = self.dist(s); a = d.sample()
        return a, d.log_prob(a).sum(-1), self.v(s).squeeze(-1)

    def evaluate(self, s, a):
        d = self.dist(s)
        return d.log_prob(a).sum(-1), d.entropy().sum(-1), self.v(s).squeeze(-1)


def collect(env, net, rng, n_ep):
    S, A, LP, R, V, DONE = [], [], [], [], [], []
    scores, errs = [], []
    for _ in range(n_ep):
        env.set_reference(random_reference(rng))
        s = env.reset(); done = False; tot = 0.0; n = 0
        while not done:
            st = torch.as_tensor(s, dtype=torch.float32)
            with torch.no_grad():
                a, lp, v = net.act(st)
            S.append(s); A.append(a.numpy()); LP.append(float(lp)); V.append(float(v))
            s, r, done, _ = env.step(a.numpy())
            R.append(r); DONE.append(done); tot += r; n += 1
        scores.append(tot/max(n, 1))
        errs.append(env.tracking_error()["chi_deg"])
    return (np.array(S, np.float32), np.array(A, np.float32), np.array(LP, np.float32),
            np.array(R, np.float32), np.array(V, np.float32), np.array(DONE),
            scores, errs)


def gae(R, V, DONE, gamma=0.97, lam=0.95):
    adv = np.zeros_like(R); last = 0.0
    for t in reversed(range(len(R))):
        nextv = 0.0 if DONE[t] else (V[t+1] if t+1 < len(V) else 0.0)
        delta = R[t] + gamma*nextv - V[t]
        last = delta + gamma*lam*(0.0 if DONE[t] else last)
        adv[t] = last
    return adv, adv + V


def train(iters=400, ep_per_iter=16, epochs=10, clip=0.2, lr=3e-4,
          ent_coef=0.01, ent_final=0.0005, vf_coef=0.5, seed=0):
    """ent_coef DECROIT au cours de l'entrainement.

    Avec un bonus d'entropie constant, sigma reste bloque (~0.60 mesure) : la
    politique garde trop de bruit pour suivre precisement. On explore beaucoup au
    debut, puis on laisse sigma diminuer pour affiner.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = TrackEnv(dt_ctrl=0.25)
    env.set_reference(random_reference(rng))
    net = ActorCritic(env.state_dim, env.action_dim)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hist = []
    for it in range(iters):
        # decroissance geometrique du bonus d'entropie
        ec = ent_coef * (ent_final/ent_coef) ** (it/max(iters-1, 1))
        S, A, LP, R, V, DONE, sc, er = collect(env, net, rng, ep_per_iter)
        adv, ret = gae(R, V, DONE)
        adv = (adv - adv.mean())/(adv.std() + 1e-8)
        S_t = torch.as_tensor(S); A_t = torch.as_tensor(A)
        LP_t = torch.as_tensor(LP)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)
        for _ in range(epochs):
            lp, ent, v = net.evaluate(S_t, A_t)
            ratio = (lp - LP_t).exp()
            l = -torch.min(ratio*adv_t,
                           torch.clamp(ratio, 1-clip, 1+clip)*adv_t).mean() \
                + vf_coef*((v-ret_t)**2).mean() - ec*ent.mean()
            opt.zero_grad(); l.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        hist.append((np.mean(sc), np.mean(er)))
        if (it+1) % 25 == 0:
            print(f"    iter {it+1:3d} | score {np.mean(sc):.3f} "
                  f"| erreur cap {np.mean(er):5.1f} deg | sigma {net.log_std.exp().mean():.2f}")
    return net, env, rng, hist


if __name__ == "__main__":
    print(f"suivi de trajectoire par melange | action continue dim {N_PRIMS}\n")
    t0 = time.time()
    net, env, rng, hist = train()
    print(f"\nentraine en {time.time()-t0:.0f}s")
    print(f"  score        : {hist[0][0]:.3f} -> {hist[-1][0]:.3f}")
    print(f"  erreur de cap: {hist[0][1]:.1f} deg -> {hist[-1][1]:.1f} deg")
    torch.save(net.state_dict(), "ppo_track.pt")

    print("\n--- evaluation sur 20 references JAMAIS VUES ---")
    errs = {"gamma": [], "chi": [], "speed": []}; var = []
    for _ in range(20):
        env.set_reference(random_reference(rng))
        s = env.reset(); done = False
        while not done:
            with torch.no_grad():
                a = net.pi(torch.as_tensor(s, dtype=torch.float32))
            s, r, done, _ = env.step(a.numpy())
        e = env.tracking_error()
        errs["gamma"].append(e["gamma_deg"]); errs["chi"].append(e["chi_deg"])
        errs["speed"].append(e["speed"])
        W = env.result()["weights"]
        var.append(float(np.abs(np.diff(W, axis=0)).sum()/max(len(W)-1, 1)))
    print(f"  erreur pente   : {np.mean(errs['gamma']):5.1f} deg")
    print(f"  erreur cap     : {np.mean(errs['chi']):5.1f} deg")
    print(f"  erreur vitesse : {np.mean(errs['speed']):5.2f} m/s")
    print(f"  variation du melange : {np.mean(var):.3f}  (0 = melange fige)")

    # --- comparaison : aleatoire / appris / oracle ---
    print("\n--- comparaison sur 20 nouvelles references ---")
    from blend import weights_over_time
    rnd, learn, orac = [], [], []
    for _ in range(20):
        K = rng.dirichlet(np.ones(N_PRIMS)*0.6, size=6)
        ref = reference_from_weights(K, 5.0, 0.02,
                                     dict(gamma=0., chi=0., mu=0., speed=5.))
        # aleatoire
        env.set_reference(ref); env.reset(); done = False
        while not done:
            _, _, done, _ = env.step(rng.normal(0, 1.5, N_PRIMS))
        rnd.append(env.tracking_error()["chi_deg"])
        # appris
        env.set_reference(ref); s_ = env.reset(); done = False
        while not done:
            with torch.no_grad():
                a = net.pi(torch.as_tensor(s_, dtype=torch.float32))
            s_, _, done, _ = env.step(a.numpy())
        learn.append(env.tracking_error()["chi_deg"])
        # oracle : on rejoue les poids qui ont genere la reference
        env.set_reference(ref); env.reset(); done = False
        while not done:
            tnow = env.k*env.dt_ctrl + env.dt_ctrl*0.5
            W_t = weights_over_time(K, np.array([tnow]), 5.0)[0]
            _, _, done, _ = env.step(np.log(np.maximum(W_t, 1e-9)))
        orac.append(env.tracking_error()["chi_deg"])
    print(f"  {'aleatoire':<12} erreur de cap {np.mean(rnd):5.1f} deg")
    print(f"  {'PPO appris':<12} erreur de cap {np.mean(learn):5.1f} deg")
    print(f"  {'oracle':<12} erreur de cap {np.mean(orac):5.1f} deg  (borne basse)")
    np.save("track_history.npy", np.array(hist))