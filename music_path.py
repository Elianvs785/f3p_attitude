"""GENERATION DU CHEMIN — uniquement a partir de la musique.

PRINCIPE (demande du superviseur)
---------------------------------
La generation ne connait AUCUNE contrainte physique. Elle produit :
  1. des WAYPOINTS relies par des LIGNES DROITES
  2. les INSTANTS de passage (les beats)
  3. un profil de ROULIS mu_ref(t), lui aussi issu de la musique

Melanger physique et generation briderait la choregraphie : on s'interdirait des
figures avant meme d'avoir essaye de les voler. C'est le SUIVI qui gere les
contraintes.

LA VITESSE N'EST PAS UN CHOIX
-----------------------------
Le chemin et les instants etant fixes, la vitesse moyenne de chaque segment vaut

    v_i = L_i / dt_i          (distance / temps disponible)

Exemple du superviseur : beats a 1 s, 2 s, 10 s avec des distances egales ->
la vitesse entre le 2e et le 3e beat est 8 fois plus lente.

L'avion F3P peut voler tres lentement (jusqu'a l'arret) en se cabrant : la
poussee le soutient, les ailes ne portent plus. C'est le regime "harrier", prevu
par le modele du superviseur (ALPHA_BETA_MAX = 89.9 deg, solve_static_hover).
"""
import numpy as np

# ------------------------------------------------------------------ chemins
def rectangle_3d(size=(6.0, 4.0, 3.0), center=(0.0, 0.0, 2.5)):
    """8 sommets d'un pave droit, dans un ordre qui en fait un circuit ferme.

    Cas de test demande par le superviseur : geometrie connue, distances fixes,
    on maitrise tout. Sert a valider le suivi avant de passer a la musique.
    """
    sx, sy, sz = np.array(size)/2.0
    c = np.array(center, float)
    # ordre : face du bas (4 coins), puis face du haut, en circuit
    V = np.array([
        [-sx, -sy, -sz], [+sx, -sy, -sz], [+sx, +sy, -sz], [-sx, +sy, -sz],
        [-sx, +sy, +sz], [+sx, +sy, +sz], [+sx, -sy, +sz], [-sx, -sy, +sz],
    ], float) + c
    return V


def cube_3d(side=4.0, center=(0.0, 0.0, 2.5), close=True):
    """Circuit sur les ARETES d'un vrai cube (cotes egaux).

    rectangle_3d donnait un pave 6x4x3 : les segments avaient 3 longueurs
    differentes, donc 3 vitesses tres differentes imposees par v = L/dt, ce qui
    melangeait deux difficultes. Ici tous les segments font la meme longueur :
    seul l'ecart entre beats fait varier la vitesse. Cas de test plus propre.

    Chemin hamiltonien sur les aretes : face du bas, une arete verticale, face
    du haut. C'est la SEULE facon de visiter 8 sommets sans diagonale.
    close=True ajoute l'arete de retour (8 segments au lieu de 7).
    """
    h = side/2.0
    c = np.array(center, float)
    V = np.array([
        [-h, -h, -h], [+h, -h, -h], [+h, +h, -h], [-h, +h, -h],
        [-h, +h, +h], [+h, +h, +h], [+h, -h, +h], [-h, -h, +h],
    ], float) + c
    return np.vstack([V, V[0]]) if close else V


def path_from_beats(beat_times, beat_force, shape="rectangle", rng=None,
                    step=1.6, z_range=(1.2, 3.8), close=True,
                    ang_min=30.0, ang_max=120.0):
    """Waypoints places a partir de la MUSIQUE seule.

    shape="rectangle" : geometrie fixe (cas de test), les beats ne donnent que
                        les INSTANTS de passage.
    shape="music"     : la direction de chaque segment vient de la musique
                        (amplitude du virage = force du beat).
    """
    bt = np.asarray(beat_times, float)
    bf = np.asarray(beat_force, float)
    bf = (bf - bf.min())/(bf.max() - bf.min() + 1e-9)

    if shape in ("rectangle", "cube"):
        # variante ajoutee, l'ancienne reste accessible (shape="rectangle")
        V = cube_3d(close=close) if shape == "cube" else rectangle_3d()
        n = len(bt)
        idx = np.round(np.linspace(0, len(V)-1, n)).astype(int) if n <= len(V) \
              else np.arange(n) % len(V)
        W = V[idx]
        return W, bt, bf

    # --- chemin issu de la musique ---
    rng = rng or np.random.default_rng(0)
    W = [np.array([0.0, 0.0, np.mean(z_range)])]
    d = np.array([1.0, 0.0, 0.0])
    for i, f in enumerate(bf[1:], start=1):
        # amplitude du virage = force du beat (le lien musical)
        # Plage d'angles : PARAMETRE de la regle musicale, pas contenu musical.
        # La musique fournit la FORCE du beat ; la convertir en degres demande
        # de choisir une echelle, comme la taille de l'arene. Un beat fort donne
        # toujours un virage plus ample qu'un beat faible : le lien musical est
        # intact, seule l'amplitude change.
        # 30-150 deg venait du QUADROTOR, qui peut pivoter sur place. Un avion
        # ne le peut pas : au-dela de ~90 deg sur des segments courts il doit
        # quasiment s'arreter, et la relance depasse la poussee disponible.
        ang = np.radians(ang_min + (ang_max - ang_min)*f)
        # plan de rotation = position dans la mesure (lien musical aussi)
        pl = np.radians(360.0*((i % 4)/4.0))
        ref = np.array([0,0,1.0]) if abs(d[2]) < 0.9 else np.array([1.0,0,0])
        s = np.cross(d, ref); s /= np.linalg.norm(s)+1e-9
        u = np.cross(d, s);   u /= np.linalg.norm(u)+1e-9
        # CHOIX DE DIRECTION plutot que clip de position (M23-M24).
        # `np.clip(p[2], z_range)` ecrasait le point contre le plafond -> PLIE
        # la geometrie -> minsnap est un solve GLOBAL (continuite jusqu'au jerk
        # sur TOUS les segments a la fois) : UN SEUL point plie corrompt
        # l'acceleration ailleurs dans le polynome, parfois loin de lui (mesure
        # : |a| plus fort LOIN du point plie que pres de lui). Meme defaut que
        # le np.clip de generator.py cote drone, meme remede : corriger la
        # direction EN AMONT, jamais la position en aval.
        #
        # 4 candidats (pl, -pl, +pi, -pi) laissaient encore 6/32 points coinces :
        # trop grossier. VERIFIE (exist.py) : a chaque etape il existe TOUJOURS
        # un plan qui respecte l'arene (0/31 cas impossibles) — il fallait une
        # recherche plus fine, pas renoncer a la contrainte.
        #
        # L'AMPLITUDE du virage (le contenu musical, = la force du beat) est
        # inchangee : on ne fait varier QUE le plan de rotation, qui code deja
        # la position dans la mesure — un choix purement geometrique.
        pls = np.linspace(0.0, 2*np.pi, 72, endpoint=False)
        best_pl, best_pen = pl, np.inf
        for cand in pls:
            perp = np.cos(cand)*s + np.sin(cand)*u
            dc = np.cos(ang)*d + np.sin(ang)*perp
            dc /= np.linalg.norm(dc)+1e-9
            zc = (W[-1] + dc*step)[2]
            pen = max(z_range[0]-zc, zc-z_range[1], 0.0)   # 0 si dans l'arene
            if pen < best_pen:
                best_pen, best_pl, best_d = pen, cand, dc
        d = best_d
        p = W[-1] + d*step
        p[2] = np.clip(p[2], z_range[0], z_range[1])   # filet, ne se declenche
        W.append(p)                                    # jamais en pratique
    return np.array(W), bt, bf


# ------------------------------------------------------------------ roulis
def roll_profile_from_music(t_grid, beat_times, beat_force,
                            strong_pct=75, hold=0.6, n_turns=1.0):
    """Profil de roulis mu_ref(t) genere par la MUSIQUE.

    Regle : un onset FORT et SOUTENU declenche un tonneau. On repere les beats
    dont la force depasse un percentile eleve, et on y insere une rotation
    complete de mu, etalee sur `hold` secondes (smoothstep pour la continuite).

    Retourne mu_ref(t) sur la grille demandee, en radians, deroule (unwrap).
    """
    bt = np.asarray(beat_times, float)
    bf = np.asarray(beat_force, float)
    bf = (bf - bf.min())/(bf.max() - bf.min() + 1e-9)
    thr = np.percentile(bf, strong_pct)
    strong = bt[bf >= thr]

    mu = np.zeros_like(t_grid)
    sign = 1.0
    for ts in strong:
        # smoothstep de 0 a 2*pi*n_turns sur [ts, ts+hold]
        u = np.clip((t_grid - ts)/max(hold, 1e-9), 0.0, 1.0)
        mu += sign * 2*np.pi*n_turns * (u*u*(3.0 - 2.0*u))
        sign = -sign            # alterne le sens pour ne pas cumuler
    return mu


def speed_reference(W, times, t_grid, blend=0.45):
    """Profil de vitesse impose par la geometrie et les instants.

    v_i = L_i / dt_i sur chaque segment, avec une transition LISSE aux jonctions.

    POURQUOI IL FAUT LISSER
    -----------------------
    Le profil brut saute aux waypoints. Or en vol lent l'avion est en regime
    harrier : sa poussee sert deja a le porter (poids 0.98 N sur 1.96 N
    disponibles), il ne reste que ~9.8 m/s2 pour accelerer. Une transition
    brutale depasse ce budget.

    Mesure du compromis (transition 0.75 -> 3 m/s) :
        blend=0.25 -> 10.0 m/s2, poussee 1.98 N, err 0.31 m : DEPASSE
        blend=0.40 ->  6.2 m/s2, poussee 1.61 N, err 0.49 m : OK
        blend=0.50 ->  5.0 m/s2, poussee 1.48 N, err 0.60 m : OK
        blend=0.70 -> 81.4 m/s2 : les fenetres de deux waypoints voisins se
                      CHEVAUCHENT et se contredisent.
    La borne est structurelle : avec w = blend*min(dt), les fenetres se touchent
    a blend = 0.5. On retient 0.45 pour garder une marge.

    L'erreur de distance introduite (0.5 m sur des segments de 6 m) se traduit
    par un leger decalage a l'arrivee. Le suivi la rattrape : le RL vise les
    WAYPOINTS, il ne suit pas aveuglement le profil de vitesse.

    DEUX APPROCHES ECARTEES
    -----------------------
    1. Lisser puis remettre chaque segment a l'echelle pour retrouver la
       distance exacte : les facteurs differant d'un segment a l'autre, on
       reintroduit des discontinuites AUX waypoints (pics a 167 m/s2).
    2. Interpoler l'abscisse curviligne par une spline cubique : les distances
       sont exactes, mais la spline DEPASSE pour raccorder des vitesses moyennes
       tres differentes (22 m/s2, pointe a 8 m/s).

    Le lissage direct, sans correction, est le meilleur compromis : l'erreur de
    distance introduite est faible et se traduit par un leger decalage a
    l'arrivee, que le suivi rattrape.
    """
    W = np.asarray(W, float); T = np.asarray(times, float)
    L = np.linalg.norm(np.diff(W, axis=0), axis=1)
    dt = np.diff(T)
    v_seg = L/np.maximum(dt, 1e-9)

    v = np.zeros_like(t_grid)
    for i in range(len(v_seg)):
        m = (t_grid >= T[i]) & (t_grid <= T[i+1])
        v[m] = v_seg[i]
    v[t_grid < T[0]] = v_seg[0]; v[t_grid > T[-1]] = v_seg[-1]

    # transition smoothstep autour de chaque waypoint interieur
    #
    # FENETRE ASYMETRIQUE (correctif). La version symetrique prenait
    #     w = blend*min(dt[i-1], dt[i])
    # donc le segment LONG n'apportait jamais sa marge : au waypoint t=10 s,
    # avec 8 s avant et 1 s apres, la transition durait 0.45 s au lieu des
    # 3.6 s disponibles. L'avion devait accelerer au maximum juste apres le
    # waypoint, dans le regime ou les ailes ne portent pas encore -> infaisable.
    # Chaque cote prend maintenant la marge de SON segment.
    #
    # La borne blend < 0.5 est inchangee : deux waypoints voisins partagent le
    # meme dt, leurs fenetres se touchent quand 2*blend*dt > dt.
    for i in range(1, len(v_seg)):
        w_av = blend*dt[i-1]          # marge du segment qui PRECEDE
        w_ap = blend*dt[i]            # marge du segment qui SUIT
        m = (t_grid > T[i]-w_av) & (t_grid < T[i]+w_ap)
        if m.sum() < 2: continue
        u = (t_grid[m] - (T[i]-w_av))/(w_av + w_ap)
        v[m] = v_seg[i-1] + (v_seg[i]-v_seg[i-1])*(u*u*(3.0-2.0*u))
    return v, v_seg