# Contrat d'architecture — chorégraphie F3P musicale

## Les trois couches (séparation stricte)
1. **Génération (musique seule)** : waypoints = beats hiérarchisés (temps forts),
   instants = beats, roulis mu_ref(t) = onsets forts et soutenus, amplitude des
   virages = force du beat. Lignes droites entre waypoints. AUCUNE physique.
2. **Suivi (RL)** : mélange continu des 4 primitives (droit, virage±, montée±,
   roulis±) recalculé à chaque instant + vitesse LIBRE. C'est lui qui porte les
   contraintes, via l'enveloppe MESURÉE (explore_limits) : taux de virage
   dépendant de la vitesse, plage de vitesse physique.
3. **Juge (solveur de Ruihan)** : verdict par le résidu (< 1e-4). Chaque jalon se
   termine par un verdict.

## Décisions figées
- La vitesse est une **action**, pas une référence : la musique contraint des
  instants discrets (être au waypoint i à T_i), la répartition entre deux
  waypoints est libre. Aucun profil v_ref dans la récompense.
- Limite de virage **dépendante de la vitesse**, calibrée sur explore_limits :
  ω_chi(v) = min(220°/s, a_lat_max/v) avec a_lat_max = 15.7 m/s²
  (mesuré : 90° en 0.5 s à 5 m/s = OK, à 7 m/s = NON). Même loi pour gamma
  (a_vert = 11.2 m/s², mesuré : verticale en 0.7 s), roulis borné 720°/s.
- Récompense à TROIS idées : ponctualité aux beats (dominante), proximité du
  chemin (potentiel de Ng et al. + terme doux), fidélité du roulis. Toutes les
  erreurs BORNÉES, toutes les observations BORNÉES (leçon des 4 échecs :
  une saturation tue un gradient).
- Un seul PPO (`ppo_core.py`), importé partout.

## Critères de réussite (mesurables, opposables)
- Cube : moyenne de ‖pos(T_i) − W_i‖ **à l'instant exact du beat** ≤ 1 m
  (pas le minimum sur une fenêtre) ; pic de taux de virage à ±0.2 s du beat.
- Musique : idem ≤ 1.5 m, résidu solveur < 1e-4 sur 100 % du vol, figure
  poids-des-primitives lisible contre les beats.
- « Tourner sur le beat » = virage CENTRÉ sur le beat (entamé avant).

## Ce qui ne bouge plus
Base du superviseur intouchée. LISdrone (Isaac quad, minsnap, timeopt, nv_rl)
= récit du rapport, pas chantier. Gel du code le 9/09.
