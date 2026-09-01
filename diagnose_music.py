import numpy as np, sys
sys.path.insert(0,'.'); sys.path.insert(0,'..'); sys.path.insert(0,'../LISdrone')
import minsnap
from music_path import path_from_beats
from f3p_attitude.solver import solve_trajectory
from f3p_attitude.constants import RESIDUAL_TOL, THRUST_MAX

bt = np.load('beat_times.npy'); bf = np.load('beat_force.npy')
step = 3.0*float(np.mean(np.diff(bt)))
W, T, F = path_from_beats(bt, bf, shape="music", step=step,
                          z_range=(1.0, 6.0), ang_min=20.0, ang_max=60.0)
C = minsnap.solve(W, T)
t = np.linspace(T[0], T[-1], 300)
P = minsnap.evaluate(C, T, t, 0)
V = minsnap.evaluate(C, T, t, 1); A = minsnap.evaluate(C, T, t, 2)
sp = np.maximum(np.linalg.norm(V, axis=1), 1e-6)
gam = np.arcsin(np.clip(V[:, 2]/sp, -1, 1))
chi = np.unwrap(np.arctan2(V[:, 1], V[:, 0]))
s = solve_trajectory(t, gam, chi, sp, np.zeros_like(t),
                     gamma_dot=np.gradient(gam, t), chi_dot=np.gradient(chi, t),
                     speed_dot=np.gradient(sp, t))
rn = np.asarray(s.residual_norm, float)
th = np.asarray(s.thrust, float)
bad = np.where(rn > RESIDUAL_TOL)[0]
ok = np.where(rn <= RESIDUAL_TOL)[0]
print(f"instants infaisables : {len(bad)}/{len(t)}  ({100*len(bad)/len(t):.0f} %)")
print()
print(f"{'grandeur':<22}{'instants OK':>16}{'instants KO':>16}")
print("-"*54)
for nom, arr in [("vitesse [m/s]", sp),
                 ("pente |gamma| [deg]", np.degrees(np.abs(gam))),
                 ("acceleration [m/s2]", np.linalg.norm(A, axis=1)),
                 ("poussee [% max]", 100*th/THRUST_MAX),
                 ("altitude [m]", P[:, 2])]:
    print(f"{nom:<22}{np.median(arr[ok]):>10.1f} (med){np.median(arr[bad]):>10.1f} (med)")
print()
print("les instants KO se produisent a t =", np.round(t[bad][:14], 1))
print()
z = P[:, 2]
print(f"altitude parcourue : {z.min():.2f} a {z.max():.2f} m  (z_range demande 1.0-6.0)")
print(f"instants KO sous 1.2 m ou au-dessus de 5.8 m : {int(((z[bad]<1.2)|(z[bad]>5.8)).sum())}/{len(bad)}")