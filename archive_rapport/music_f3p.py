"""MUSIQUE -> FIGURES F3P.

L'aboutissement : les beats de la chanson pilotent la choregraphie d'un avion
F3P, et la faisabilite est verifiee par le modele aerodynamique.

PRINCIPE
--------
Les POINTS DE CONTROLE du melange sont places AUX INSTANTS DES BEATS. Entre deux
beats, l'interpolation smoothstep fait evoluer le melange en douceur. Le
changement de manoeuvre tombe donc SUR le beat par construction — meme garantie
que pour le drone (invariant "1 beat = 1 changement").

La FORCE du beat choisit l'intensite :
    beat fort   -> manoeuvre franche (montee, tonneau, virage serre)
    beat moyen  -> virage marque
    beat faible -> inflexion douce, ou ligne droite

CE QUI EST REUTILISE
--------------------
  extract_beats / beat_force      (pipeline musical existant)
  blend                            (melange continu)
  f3p_attitude.solve_trajectory    (validation physique reelle)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from blend import blend, PRIMS, N_PRIMS

I = {p: PRIMS.index(p) for p in PRIMS}


def w(**kw):
    v = np.zeros(N_PRIMS)
    for k, x in kw.items():
        v[I[k]] = x
    return v


# --- vocabulaire de manoeuvres, classe par intensite ---
# (une manoeuvre = un melange de primitives ; le signe alterne pour varier)
GENTLE = [lambda s: w(straight=.6, **{f"turn_{s}": .4}),
          lambda s: w(straight=.7, climb_up=.3),
          lambda s: w(straight=.7, climb_down=.3)]
MEDIUM = [lambda s: w(**{f"turn_{s}": .8}, straight=.2),
          lambda s: w(climb_up=.6, **{f"turn_{s}": .4}),
          lambda s: w(climb_down=.6, **{f"turn_{s}": .4}),
          lambda s: w(**{f"roll_{s}": .6}, straight=.4)]
STRONG = [lambda s: w(**{f"turn_{s}": 1.0}),
          lambda s: w(climb_up=1.0),
          lambda s: w(climb_down=1.0),
          lambda s: w(**{f"roll_{s}": .9}, accel=.1),
          lambda s: w(climb_up=.6, **{f"roll_{s}": .4})]


def load_beats(path_times="beat_times.npy", path_force="beat_force.npy",
               fallback_bpm=112.0, fallback_dur=20.0, rng=None):
    """Charge les beats de la chanson, ou en fabrique si les fichiers sont absents."""
    if os.path.exists(path_times) and os.path.exists(path_force):
        t = np.load(path_times); f = np.load(path_force)
        src = "chanson"
    else:
        rng = rng or np.random.default_rng(0)
        step = 60.0/fallback_bpm
        t = np.arange(0.0, fallback_dur, step)
        f = 0.25 + 0.75*np.abs(np.sin(np.arange(len(t))*0.9)) * rng.uniform(.7, 1.3, len(t))
        src = f"synthetique ({fallback_bpm:.0f} BPM)"
    f = np.asarray(f, float)
    f = (f - f.min())/(f.max() - f.min() + 1e-9)      # force normalisee 0..1
    return np.asarray(t, float), f, src


def choreograph(beat_times, beat_force, rng, min_gap=0.35, max_beats=None,
                phrase=2):
    """Associe une manoeuvre a chaque beat selon sa FORCE.

    Retourne (control_weights, control_times, choix) : les poids aux beats.
    """
    # on ecarte les beats trop rapproches (l'avion n'a pas le temps de manoeuvrer)
    keep = [0]
    for i in range(1, len(beat_times)):
        if beat_times[i] - beat_times[keep[-1]] >= min_gap:
            keep.append(i)
    keep = np.array(keep)
    if max_beats:
        keep = keep[:max_beats]
    T = beat_times[keep]; F = beat_force[keep]

    # etat approximatif suivi au fil de la choregraphie, pour rester compact :
    #  - on force le retour du roulis quand il s'est trop accumule
    #  - on privilegie les virages du cote qui ramene vers le centre
    K = []; choix = []; sign = "right"; last_family = None
    mu_acc = 0.0            # roulis cumule (deg)
    chi_acc = 0.0           # cap cumule (deg)
    for i, (t, f) in enumerate(zip(T, F)):
        if f < 0.33:
            fam, name = GENTLE, "doux"
        elif f < 0.66:
            fam, name = MEDIUM, "moyen"
        else:
            fam, name = STRONG, "fort"
        # eviter de repeter exactement la meme manoeuvre
        j = int(rng.integers(len(fam)))
        if last_family == (name, j) and len(fam) > 1:
            j = (j + 1) % len(fam)
        last_family = (name, j)

        # --- garder la choregraphie compacte et equilibree ---
        # le sens du virage est choisi pour ne pas partir toujours dans la meme
        # direction (sinon la trajectoire s'etire au lieu de tourner)
        if abs(chi_acc) > 200.0:
            sign = "left" if chi_acc > 0 else "right"
        # si le roulis s'est trop accumule, on impose un tonneau inverse
        if abs(mu_acc) > 400.0:
            wv = w(**{f"roll_{'left' if mu_acc > 0 else 'right'}": .9, "straight": .1})
            K.append(wv); choix.append((float(t), float(f), name, -1,
                                        "left" if mu_acc > 0 else "right"))
            mu_acc += (-360.0 if mu_acc > 0 else 360.0)
            sign = "left" if sign == "right" else "right"
            continue

        wv = fam[j](sign)
        K.append(wv)
        choix.append((float(t), float(f), name, j, sign))
        # mise a jour approximative des cumuls (duree ~ intervalle entre beats)
        dtb = float(T[i+1]-T[i]) if i+1 < len(T) else 0.5
        chi_acc += np.degrees(wv[I[f"turn_{sign}"]] * (75.0*np.pi/180) * dtb) \
                   * (1 if sign == "right" else -1)
        mu_acc += np.degrees(wv[I[f"roll_{sign}"]] * (240.0*np.pi/180) * dtb) \
                  * (1 if sign == "right" else -1)
        # --- le sens ne change PAS a chaque beat ---
        # Alterner a chaque beat fait s'annuler les virages : le cap total reste
        # nul et la trajectoire s'etire en ligne droite (mesure : 45 m en x pour
        # 4 m en y). Un pilote garde le meme sens pendant une PHRASE musicale
        # (une phrase), ce qui trace des boucles restant dans le volume.
        # Mesure du ratio x/y de l'encombrement selon la longueur de phrase :
        #     1 temps -> 10.7 (etire)   2 temps -> 0.4 (compact)
        #     4 temps ->  2.3           8 temps -> 0.8
        # -> phrase = 2 retenue.
        if (i + 1) % phrase == 0 or abs(chi_acc) > 270.0:
            sign = "left" if sign == "right" else "right"
            chi_acc = 0.0
    # on termine en ligne droite
    K.append(w(straight=1.0)); T = np.append(T, T[-1] + 0.6)
    return np.array(K), T, choix


def build(beat_times, beat_force, rng, speed0=5.0, dt=0.02, max_beats=None, phrase=2):
    K, T, choix = choreograph(beat_times, beat_force, rng, max_beats=max_beats, phrase=phrase)
    duration = float(T[-1] - T[0])
    seq = blend(K, duration, dict(gamma=0., chi=0., mu=0., speed=speed0), dt)
    seq["beat_times"] = T - T[0]
    seq["choix"] = choix
    seq["K"] = K
    seq["name"] = "musique"
    seq["info"] = f"{len(choix)} beats -> manoeuvres"
    return seq


def positions(seq):
    t, g, c, s = seq["t"], seq["gamma"], seq["chi"], seq["speed"]
    v = np.stack([s*np.cos(g)*np.cos(c), s*np.cos(g)*np.sin(c), s*np.sin(g)], 1)
    p = np.zeros_like(v)
    for i in range(len(t)-1):
        p[i+1] = p[i] + 0.5*(v[i]+v[i+1])*(t[i+1]-t[i])
    return p
