# Journal de mesures — partie fixed-wing

Tout chiffre ici a ete produit par un script du depot, pas estime.
Juge de faisabilite : **residu de l'equilibre des forces < 1e-4 N**
(et non `r.success`, qui ne dit que la convergence de l'optimiseur ; et non la
poussee, bornee par construction dans le solveur donc jamais depassee).

---

## M1. Diagnostic de l'infaisabilite — ablation
`diagnose_residual.py`, cas de test du superviseur (beats 0,1,2,10,11,12,13.5,15 s).

| cas | residu max | instants KO | poussee max |
|---|---|---|---|
| A. baseline | 8.84e-02 | 12/751 | 100 % |
| B. sans acceleration longitudinale | 2.38e-08 | 0/751 | 81 % |
| C. sans rotation | 5.82e-03 | 4/751 | 100 % |
| D. sans les deux | 1.52e-08 | 0/751 | 78 % |

Lecture : supprimer l'acceleration longitudinale suffit, supprimer la rotation
non. L'acceleration est presente dans **tous** les echecs.

Contre-exemple instructif : a t = 1.02 s la figure exige a_transverse =
18.9 m/s2 et reste **volable**, alors qu'a t = 11.1 s elle echoue avec
13.1 m/s2. A 6 m/s les ailes portent et fournissent la force laterale ; a 5 m/s
en sortie de regime lent l'avion est encore cabre et la poussee fait tout.

> **Une acceleration modeste devient impayable quand les ailes ne travaillent
> pas encore.** Ce n'est pas la vitesse basse qui coute (les 8 s a 0.75 m/s
> passent sans probleme), c'est le CHANGEMENT de vitesse en sortie de vol lent.

## M2. Correctif 1 — fenetre de lissage asymetrique
`music_path.py::speed_reference`. La fenetre valait `blend*min(dt[i-1], dt[i])`,
donc le segment long n'apportait jamais sa marge : au waypoint t = 10 s, avec
8 s avant et 1 s apres, la transition durait 0.45 s au lieu des 3.6 s
disponibles. Chaque cote prend maintenant la marge de son propre segment.

| | instants KO | fenetre fautive |
|---|---|---|
| avant | 12/751 | t = 11.00 -> 12.10 s |
| apres | 5/751 | t = 11.02 -> 11.10 s |

Borne structurelle inchangee : `blend < 0.5`, sinon deux fenetres voisines
partagent le meme dt et se chevauchent.

Effet secondaire mesure : l'ecart moyen au chemin de l'oracle passe de
**0.73 m a 0.52 m** sans toucher au suivi.

Les 5 instants restants ne sont pas un defaut de code : les waypoints a 11 s et
12 s sont a 1 s l'un de l'autre des deux cotes, il n'y a aucune marge a
recuperer. La demande (3 -> 6 m/s en 1 s dans un virage) est reelle.

## M3. Correctif 2 — temps calcule et non accumule
`track_f3p_env.py::step`. `self.t += self.dt` repete 750 fois avec dt = 0.02
(non representable en binaire) laissait **2.3e-13 s** d'erreur accumulee :
`T[-1] <= self.t` renvoyait False et le **dernier waypoint ne recevait jamais sa
recompense de synchronisation**. Le temps est maintenant calcule depuis le
numero de sous-pas.

| | franchissements enregistres | wp_err oracle |
|---|---|---|
| avant | 6/7 | 0.96 m |
| apres | 7/7 | 1.01 m |

La metrique EMPIRE en corrigeant : elle omettait le waypoint le plus mal
negocie. C'est le comportement attendu d'une correction honnete.

## M4. La musique reelle
`beat_times.npy` / `beat_force.npy`, 32 beats, 17.9 s, 104 BPM.

| grandeur | valeur |
|---|---|
| intervalle min / median / max | 0.511 / 0.580 / 0.604 s |
| rapport max/min | **1.18** |
| force onset min / max | 0.67 / 23.61 |

**Le cas motivant du superviseur (beats a 1, 2, 10 s -> rapport de vitesse 8x)
ne se presente pas dans la musique reelle.** Le tempo est regulier. Le travail
de M1-M2 valide donc la ROBUSTESSE d'un cas qui n'arrive pas ; il n'etait pas
urgent, il reste correct.

Le probleme reel est autre : 0.58 s entre beats, alors qu'un virage a 90 deg a
220 deg/s prend deja 0.41 s. En prenant tous les beats comme waypoints, l'avion
est en virage permanent et aucune ligne droite n'est lisible.

## Etat des baselines sur le cube (8 sommets, 15 s, 150 pas de controle)

| politique | score/pas | ecart chemin | erreur waypoint | erreur roulis |
|---|---|---|---|---|
| aleatoire | +0.448 | 15.13 m | 19.24 m | — |
| oracle (poursuite pure) | +1.274 | 0.52 m | 1.01 m | 7 deg |
| RL (PPO) | *a mesurer* | | | |

## Limites connues a declarer
- Le solveur fait un bilan de **forces** seulement : ni moments, ni inertie de
  roulis, ni debattement d'ailerons. Faire tourner mu ne coute donc rien et la
  vitesse de roulis doit etre bornee a la main (720 deg/s).
- Le solveur repond "ces commandes existent", pas "un controleur peut les
  tenir".

## M5. Echec du RL sur le cube — gradient nul, pas faille exploitable
`ppo_cube.py`, 150 iterations, 69 s.

| politique | score/pas | ecart chemin | erreur waypoint | erreur roulis |
|---|---|---|---|---|
| aleatoire | 0.448 | 15.13 m | 19.24 m | 24 deg |
| oracle | 1.274 | 0.52 m | 1.01 m | 6 deg |
| RL (PPO) | 0.741 | 12.60 m | 17.88 m | **6 deg** |

Le RL egale l'oracle sur le ROULIS et n'apprend presque rien sur la position.

`diagnose_reward.py` — decomposition de la recompense :

| terme | part du gain accessible |
|---|---|
| position | 69.9 % |
| roulis | 16.9 % |
| vitesse | 9.2 % |
| synchronisation | 3.9 % |

La position vaut 70 % du gain, donc son POIDS n'est pas en cause. C'est sa
LOCALISATION : toute la valeur tient dans les deux metres autour du chemin, et
l'agent demarre a 13.9 m de mediane.

| ecart au chemin | gain a se rapprocher de 1 m |
|---|---|
| 1 m | 7.8e-01 |
| 5 m | 1.9e-03 |
| 12 m | 5.3e-08 |

La ou l'agent se trouve, le gradient local est ~1e6 fois plus fort sur le
roulis. **Il n'ignore pas la position, il ne la sent pas.**

> Distinction a garder pour le rapport : les trois echecs de reward shaping
> precedents venaient d'une FAILLE EXPLOITABLE (l'agent trouve le chemin le
> moins couteux). Celui-ci vient d'un GRADIENT NUL. Deux categories
> differentes, deux remedes differents.

## M6. Correctif 3 — shaping par potentiel
Ng, Harada & Russell (1999) : ajouter `F = gamma*Phi(s') - Phi(s)` ne change pas
la politique optimale, quelle que soit Phi. On accelere l'apprentissage sans
dicter le comportement — ce qui manquait aux corrections precedentes.

    Phi(s) = -( longueur restante le long du chemin + K_LAT * ecart lateral )

Trois choix, chacun pour une raison mesuree :
- **abscisse curviligne** et non waypoint vise : le waypoint vise change avec
  l'HORLOGE, le potentiel sauterait a chaque beat et paierait le simple
  ecoulement du temps.
- **gamma = 1** dans le shaping (GAE utilise 0.97) : avec gamma < 1,
  `F = k*(rem - gamma*rem')` vaut `0.03*k*rem` meme sans progresser, ce qui
  recompenserait le fait d'etre loin de l'arrivee.
- **ecart lateral dans le potentiel** : sans lui, l'abscisse curviligne avance
  aussi en volant PARALLELE au chemin a 15 m. L'agent toucherait toute la
  progression sans jamais se rapprocher.

| a 13 m du chemin, se rapprocher de 1 m rapporte | |
|---|---|
| ancien terme `exp(-1.5*e_lat)` | +4.8e-06 |
| nouveau terme de potentiel | **+1.000** |

Separation oracle / aleatoire : 0.777 -> **1.092** par pas.
Metriques de l'oracle inchangees (0.52 m) : on a change ce que l'agent PERCOIT,
pas ce que l'environnement FAIT.

## M7. Le shaping ne suffit pas — le probleme etait la PARAMETRISATION
Relance apres M6 : ecart au chemin **16.09 m** (contre 12.60 m avant le
shaping). Le correctif de recompense n'a rien apporte. On arrete de retoucher la
recompense et on regarde la structure.

Indice ignore jusque-la : **sigma reste a 0.58** sur trois entrainements
successifs, alors qu'il part de 0.61. La politique ne se resserre jamais. Ce
n'est pas un defaut de signal, c'est une politique qui n'a pas les MOYENS d'etre
precise.

Autorite de commande reellement accessible, `w = softmax(logits)` :

| logits | \|chi_dot\| median | p95 |
|---|---|---|
| N(0, 0.6) — le regime observe | **7.1 %** | 25.0 % |
| N(0, 2.0) | 10.4 % | 76.8 % |
| ORACLE (ce qu'il faut) | **52.7 %** | 100 % |

Ecart de logits necessaire : 1.9 pour 50 % du taux max, 4.0 pour 90 %. La
politique part de logits nuls, et le **bonus d'entropie penalise justement les
distributions piquees**. On lui demandait de faire emerger du bruit une
structure que son objectif combat.

> Le RL n'a pas echoue a apprendre a virer. Il n'en avait pas les moyens.

## M8. Correctif 4 — taux directs au lieu du softmax
`oracle.py` avait deja tranche : *"Une premiere version notait chaque primitive
puis appliquait un softmax. Mauvais [...] Ici on construit les poids
directement."* L'environnement, lui, etait reste au softmax : oracle et env
n'utilisaient pas la meme parametrisation.

La politique sort maintenant 3 nombres dans [-1,1] (fraction du taux max sur
chaque axe) + la modulation de vitesse. Les poids sont reconstruits exactement
comme dans `pursuit_weights`. Ancien mode conserve via
`TrackF3P(action_mode="softmax")`.

Le cadrage du superviseur est intact : 4 primitives, poids toujours calcules et
affichables, "40 % droit / 60 % virage" toujours lisible. Seule la
parametrisation change — **lineaire au lieu qu'exponentielle**.

| | action_dim | oracle : ecart chemin | \|chi_dot\| median sous bruit 0.6 |
|---|---|---|---|
| softmax | 8 | 0.52 m | 7.1 % |
| taux directs | **4** | **0.51 m** | **30.0 %** |

L'oracle donne le meme resultat dans les deux modes : la reparametrisation est
equivalente, on n'a pas change ce que l'environnement PEUT faire, seulement ce
que la politique peut ATTEINDRE.

## M9. Refonte v2 (track_music_env) — validation numpy avant entrainement
Fondations de track_blend_env (tout borne) + visee + ponctualite aux beats +
vitesse LIBRE + enveloppe omega(v) = min(220 deg/s, 15.7/v) calibree sur
explore_limits. Cube du superviseur :

| politique | beat_err moyen | max | chemin | pic virage vs beat |
|---|---|---|---|---|
| aleatoire | 29.56 m | 50.0 m | 22.4 m | — |
| oracle (poursuite + v=dist/t_left) | **1.13 m** | 2.71 m | 1.05 m | 0.36 s |

Le pire beat de l'oracle est le coin aborde a 6 m/s (omega(6)=150 deg/s, rayon
2.3 m) : il ne ralentit pas avant le coin. Le RL voit l'angle du virage a venir
et decide la vitesse -> l'ecart au critere (<= 1 m) est exactement la marge
laissee a l'apprentissage.

## M10. Le RL vole la bonne trajectoire au mauvais moment
Cube v2, 400 iterations :

| politique | score | beat_err | chemin | roulis |
|---|---|---|---|---|
| oracle | 0.948 | 1.13 m | 1.05 m | 7 deg |
| RL (PPO) | 0.904 | **4.37 m** | **0.70 m** | 7 deg |

Le RL suit le chemin MIEUX que l'oracle et rate les waypoints 4x plus. Probleme
de synchronisation pur, pas de pilotage.

Budget de recompense mesure sur un episode (oracle, 150 pas) :

| terme | total | part | declenchements |
|---|---|---|---|
| ponctualite | 10.8 | **8 %** | 7 fois |
| chemin | 57.1 | 40 % | 150 fois |
| potentiel | 17.5 | 12 % | 150 fois |
| roulis | 56.7 | 40 % | 150 fois |

> Le terme declare DOMINANT par le contrat pesait 8 %. Erreur d'unites :
> j'ai compare des COEFFICIENTS sans regarder les FREQUENCES (par-evenement
> contre par-pas). L'agent a optimise ce qu'on payait, et tres bien.

## M11. Correctif — rebalance + potentiel d'horaire
`W_TIME` 3 -> 20, `W_MU` 0.4 -> 0.25, et ajout d'un potentiel d'horaire
`Phi = -|avance parcourue - avance prevue|` (7 evenements sur 150 pas restent
sparses : avec gamma=0.97 le credit ne remonte que ~30 pas, et l'ecart entre les
beats a t=2 s et t=10 s en fait 80). L'agent observe aussi son retard, sinon il
ne peut pas le corriger.

| terme | part apres |
|---|---|
| ponctualite | **40 %** |
| chemin | 31 % |
| roulis | 19 % |
| potentiel position | 10 % |
| potentiel horaire | 0 % |

Le potentiel d'horaire totalise 0.0 sur l'oracle : c'est la propriete du
theoreme (la somme telescope en Phi(fin)-Phi(debut), et l'oracle part et finit
a l'heure). Il ne paie pas celui qui est deja synchronise, il pousse celui qui
derive — un guide, pas une prime.

Controle : oracle inchange a 1.13 m. On n'a modifie que ce que l'agent PERCOIT.

## M12. Le "cube" n'en etait pas un
`rectangle_3d` produisait un PAVE 6x4x3. Les 7 segments etaient bien des aretes
(angles tous a 90 deg, aucune diagonale) mais de 3 longueurs differentes, donc
`v = L/dt` melangeait DEUX causes de variation de vitesse : la longueur du
segment ET l'ecart entre beats. L'arete de fermeture n'etait jamais parcourue.

`cube_3d(side=4, close=True)` : 8 segments de 4.0 m, circuit ferme, 9 waypoints.
Toutes les longueurs egales -> seul l'ecart entre beats fait varier la vitesse.
C'est l'exemple du superviseur isole proprement.

Vitesses imposees : 4.0, 4.0, **0.5**, 4.0, 4.0, 2.67, 2.67, 2.67 m/s
(0.5 m/s = la borne basse du vol harrier, sur le silence de 8 s).

| politique | beat_err | max | chemin | roulis |
|---|---|---|---|---|
| aleatoire | 32.47 m | 52.9 m | 25.3 m | 94 deg |
| oracle | **0.60 m** | 2.49 m | 0.49 m | 12 deg |

**Critere du CONTRAT (<= 1 m) ATTEINT par l'oracle sur le vrai cube.**
Seuls 2 beats depassent : le beat 4 (2.49 m) est la SORTIE DU SEGMENT LENT —
accelerer de 0.5 a 4 m/s hors du regime harrier, exactement l'infaisabilite
diagnostiquee par ablation en M1. Le meme phenomene physique revient au meme
endroit par un chemin different.

A verifier : pendant les 8 s lentes, le melange montre turn_right proche de
100 % et le trace fait de petites boucles — l'oracle tourne en rond pour tuer le
temps au lieu de voler droit lentement. Myope : c'est la marge du RL.

## M13. Le cube etait involable, pas mal suivi
La trace ne ressemblait a rien : ni oracle ni RL ne dessinaient le cube. Un
oracle ET une politique apprise qui echouent de la MEME facon geometrique
designent l'environnement, pas l'apprentissage.

Experience discriminante — carre PLAN (aucune arete verticale) :

| cas | beat_err | longueur volee / reference | gamma max |
|---|---|---|---|
| carre plan, 4 m/s | **0.07 m** | 111 % | 0 deg |
| carre plan, 2 m/s | 0.00 m | 105 % | 0 deg |
| cube, calendrier (1,2,10 s) | 2.51 m | **158 %** | 59 deg |

Le carre est PARFAIT : le controleur sait dessiner un carre net, les 11 % de
longueur en trop sont l'arrondi des coins. Le cube vole 18 m de trop sur 32.

Balayage du calendrier sur le meme cube :

| calendrier | beat_err | longueur |
|---|---|---|
| superviseur (1,2,10 s) | 2.51 m | 158 % |
| regulier 1.0 s/arete (4 m/s) | 2.18 m | 134 % |
| regulier 1.5 s/arete (2.7 m/s) | 0.51 m | 121 % |
| **regulier 2.0 s/arete (2 m/s)** | **0.01 m** | **113 %** |
| regulier 3.0 s/arete (1.3 m/s) | 0.00 m | 107 % |

**La geometrie n'etait pas en cause, le calendrier l'etait.** Deux calculs
d'une ligne le montrent :

1. **Arete verticale en 1 s a 4 m/s.** Cabrer de 90 deg prend 90/160 = 0.56 s a
   omega_gam(4) = 160 deg/s. Pendant ce cabre on parcourt deja 2.2 m d'une arete
   de 4 m. Aucun pilote ne peut suivre cette arete a ce rythme.
2. **Rapport 4.0 -> 0.5 m/s.** Deceleration a 5 m/s2 : 0.70 s, soit 1.6 m
   parcourus. On depasse donc le waypoint AVANT d'avoir ralenti, puis on tourne
   en rond a 0.5 m/s (rayon 13 cm) pour tuer les 8 s. Ce sont les petites
   boucles observees sur la trace.

> A dire au superviseur : avec des aretes de longueur FIXE, son calendrier
> impose un rapport de vitesse de 8. Si l'espacement des waypoints suivait
> l'intervalle entre beats — regle purement musicale, aucune physique dans la
> generation — le cas disparaitrait par construction. Et sur la vraie musique
> (rapport 1.18) il ne se presente pas.

## M14. CORRECTION de M13 — le calendrier n'etait pas involable
M13 concluait que le calendrier extreme du superviseur (beats a 1, 2, 10 s)
rendait le cube physiquement involable. **C'etait faux.** Il etait involable
POUR UN CONTROLEUR QUI VISE DES POTEAUX FIXES.

Le calcul de M13 ("cabrer prend 0.56 s, on a deja parcouru 2.2 m") etait juste
sur les chiffres et faux sur la conclusion : il supposait que le cabre commence
AU waypoint. Rien ne l'impose.

## M15. Suivi du POINT MOBILE (formulation du superviseur)
La reference n'est pas le waypoint W_i mais un point qui glisse le long des
lignes droites et se trouve sur W_i exactement a l'instant T_i. L'agent vise ce
point, legerement en avance (`t_lead = 0.35 s`, pure pursuit).

Trois choses se reglent d'un coup :
- **le depassement disparait par construction** : on ne depasse pas une cible
  qui avance devant soi ;
- **position et horaire cessent d'etre concurrents** : ||pos - r(t)|| contient
  les deux, puisque etre au bon endroit au bon moment EST la meme chose ;
- **la recompense se reduit** a un terme dominant + le roulis (au lieu des
  cinq termes que j'avais deja mal ponderes deux fois).

Oracle, meme geometrie, memes instants :

| calendrier | controleur | beat_err | suivi | longueur volee |
|---|---|---|---|---|
| regulier 2 s | waypoint | 0.01 m | — | 113 % |
| regulier 2 s | **reference** | 0.20 m | 0.28 m | **106 %** |
| STRESS (1,2,10 s) | waypoint | 2.51 m | — | 158 % |
| STRESS (1,2,10 s) | **reference** | **0.38 m** | 0.28 m | **104 %** |

Sur le cas extreme : erreur divisee par 6.6, longueur volee 158 % -> 104 %.
Le pire beat passe de 9.66 m a 0.95 m.

Piege rencontre en chemin (a garder) : premiere version de la loi de vitesse
`v = v_ref + 1.5*e` avec `e = ||pos - r(t)||`, une NORME donc toujours positive.
L'avion ne pouvait qu'accelerer, prenait de l'avance et devait boucler pour
attendre la reference -> **193 %** de longueur volee. La correction doit etre
SIGNEE : `v = v_ref - 1.2*lag` ou `lag` est l'ecart LE LONG du chemin.

Ancien mode conserve : `TrackMusic(follow="waypoint")`, et la comparaison des
deux controleurs est une figure du rapport.

## M16. Pourquoi l'ancien env marchait et pas celui-ci
Question posee : le PPO suivait wingover / huit / slalom tres bien, pourquoi pas
une ligne brisee ? Deux raisons, la seconde etant un manque de ma part.

**1. L'ancienne reference etait ATTEIGNABLE PAR CONSTRUCTION.** Le docstring de
`track_blend_env.reference_from_weights` le dit : les figures etaient PRODUITES
par le modele de melange lui-meme. Une solution exacte existait dans l'espace
d'action ; PPO n'avait qu'a la retrouver. Une ligne brisee a des coins
instantanes : aucune sequence de poids ne la realise. L'agent cherche un
compromis, pas une solution.

**2. L'ancien env DONNAIT l'attitude exigee, le mien la faisait deviner.**
L'ancienne observation contenait gamma_ref, chi_ref, speed_ref, maintenant et a
l'horizon. La mienne ne contenait que des positions : l'agent devait inferer
quelle attitude les produit, alors que l'information existe et etait jetee.

Correctif : ajout de la direction et de la vitesse EXIGEES par la reference, a
t et a t + t_lead (8 composantes). `state_dim` 24 -> 32, toutes bornees
(max |obs| = 1.87). Oracle inchange : la reparametrisation ne change que ce que
l'agent PERCOIT.

## M17. Le plancher physique du suivi — l'oracle y est deja
Balayage de l'anticipation (oracle, calendrier STRESS) :

| t_lead | distance visee a 4 m/s | beat_err |
|---|---|---|
| 0.05 s | 0.20 m | 0.80 m |
| 0.20 s | 0.80 m | 0.53 m |
| **0.35 s** | 1.40 m | **0.38 m** |
| 0.50 s | 2.00 m | 0.61 m |

t_lead = 0.35 est deja l'optimum : plus court, l'avion reagit trop tard ; plus
long, il coupe. Ce n'etait pas le coupable.

Plancher physique — un virage a 90 deg de rayon R ecarte de R*(sqrt(2)-1) :

| vitesse | rayon | ecart minimal au coin |
|---|---|---|
| 0.5 m/s | 0.13 m | 0.05 m |
| 2.0 m/s | 0.52 m | 0.22 m |
| 4.0 m/s | 1.04 m | **0.43 m** |

> L'oracle est a 0.38 m de moyenne : **il est AU plancher physique**. Le suivi de
> ligne fonctionne. "Ca coupe" n'est pas une anomalie, c'est l'arrondi minimal
> impose par le rayon de virage. Le RL, lui, est a 3-4x ce plancher.

Cible realiste pour le RL sur ce cas : ~0.4-0.6 m, pas zero.

## M18. Reference minsnap + RL residuel
**x(t) minsnap** (LISdrone/minsnap_v2, memes waypoints, memes beats) : C2 —
position, vitesse, acceleration continues et FINIES (a_max 6.35 m/s2 pour
19.6 disponibles). La ligne brisee est C0 : acceleration INFINIE aux coins,
reference inatteignable par principe (cause n.1 de M16).

| calendrier | oracle polyline | oracle minsnap |
|---|---|---|
| regulier 2 s | 0.20 m | **0.04 m** |
| STRESS (1,2,10 s) | **0.36 m** | 1.78 m |

Chaque reference a son domaine : minsnap 5x meilleur des que le calendrier est
raisonnable (= la vraie musique), polyline plus sur sur les calendriers
pathologiques (la bissectrice imposee + 8 s pour 4 m font divaguer le
polynome). Le defaut suit la mesure : polyline si STRESS, minsnap sinon.

**RL residuel** (Silver et al. ; Johannink et al.) : a = oracle(s) + delta.
Motif des 5 runs from scratch : bruitee > moyenne, sigma bloque, degradation en
fin de run, greedy qui s'effondre sur un beat. Cause : l'oracle est une fonction
LINEAIRE de grandeurs observees — on demandait a PPO de redecouvrir une formule
qu'on possede. En residuel : depart A l'oracle (verifie : delta=0 -> 1.78 m =
oracle au centieme), exploration locale (log_std initial -1.4), le reseau
n'apprend que ce que la poursuite myope ne sait pas faire.

## M19. JALON B ATTEINT — le RL bat l'oracle sur le cas dur
Reference minsnap + RL residuel, 300 iterations (~11 min).

| cas | politique | beat_err | suivi | chemin |
|---|---|---|---|---|
| regulier + minsnap | oracle | 0.05 m | 0.11 m | 0.21 m |
| regulier + minsnap | RL | 0.09 m | 0.10 m | 0.17 m |
| **STRESS + polyline** | oracle | 0.53 m | 0.38 m | 0.27 m |
| **STRESS + polyline** | **RL** | **0.45 m** | **0.23 m** | **0.13 m** |

Sur le cas facile les deux sont au bruit : il n'y avait rien a apprendre.
Sur le cas dur **le RL bat l'oracle sur les trois metriques** (suivi -40 %,
ecart au chemin -50 %) — exactement la ou la poursuite est myope.

Le residuel demarre AU niveau de l'oracle des la premiere iteration (beat_err
0.21 m a l'iteration 30, contre 41 m from scratch). Les 5 runs from scratch
plafonnaient a 1.4-2.2 m.

## M20. VERDICT DU SOLVEUR — la trajectoire n'est PAS volable
`verify_flight.py`, cube regulier + minsnap, oracle :

```
poussee    : 0.00 - 1.33 N (68 % du max)
alpha      : -19 a 90 deg (la butee)
residu max : 6.62e-01   (tolerance 1e-04)
VERDICT    : INFAISABLE — 29/267 instants, fenetre t = 14.2 a 15.1 s
```

t = 14-16 s est le segment 7->8 : **l'arete verticale DESCENDANTE**.

## M21. Pourquoi : montee et descente ont des contraintes OPPOSEES
Balayage au solveur, vol vertical a vitesse constante (poids = 0.98 N) :

| vitesse | PIQUE (gamma = -90) | MONTEE (gamma = +90) |
|---|---|---|
| 2 m/s | non | **VOLABLE** |
| 4 m/s | non | **VOLABLE** |
| 6 m/s | **VOLABLE** | non |
| 7 m/s | **VOLABLE** | — |

- **En piqué**, seule la TRAINEE peut retenir l'avion contre son poids : il faut
  aller VITE pour en generer assez. Minimum ~6 m/s.
- **En montee**, la poussee doit vaincre le poids PLUS la trainee : il faut
  aller LENTEMENT pour que la trainee reste faible. Maximum ~4 m/s.

> Le cube a une arete montante et une arete descendante, et un calendrier
> regulier leur impose la MEME vitesse. **Aucune vitesse unique ne satisfait les
> deux.** La trajectoire etait involable par construction, quel que soit le
> pilote.

Correction : les deux aretes verticales doivent etre parcourues en des TEMPS
differents — decision de la couche musique, pas du suivi. A verifier sur la
vraie chanson : un pique exactement vertical y est un cas de mesure nulle,
il n'apparaitra probablement pas.

Limite a redire : le solveur repond "ces commandes existent", pas "un
controleur peut les tenir" (bilan de FORCES seulement, ni moments ni inertie).

## M22. La musique reelle — la faisabilite ne vient pas d'ou on croyait
Test decisif : on juge la trajectoire IDEALE (minimum snap par les waypoints aux
beats) DIRECTEMENT au solveur, sans aucun controleur. C'est la borne superieure
absolue : si elle echoue, la faute est a la geometrie musicale ; si elle passe,
la faute est au suivi.

| beats retenus | angle max | infaisable |
|---|---|---|
| tous (32) | 60 deg | **3-4 %** |
| forts+moyens (21) | 60 deg | 11 % |
| forts seuls (11) | 60 deg | 22 % |

**Moins de beats est PIRE.** Avec un `step` fixe, retirer des beats allonge
l'intervalle sans allonger le segment : la vitesse tombe a ~1 m/s, l'aile ne
porte plus, et la force laterale du virage doit venir de la poussee. Corrige en
faisant suivre l'espacement a l'intervalle (regle musicale), l'ecart se reduit
mais ne disparait pas.

Hypotheses TESTEES ET REJETEES (chacune ~10 min) :
- *les vitesses imposees `v_nom*cos(delta/2)` de minsnap_v2* — vitesses libres :
  meme resultat (22 % vs 20 %).
- *l'inclinaison mu = 0 interdisait de tourner* — avec un virage coordonne
  calcule depuis l'acceleration : **aucun changement** (3 % vs 3 %). Le modele
  est une plaque plane sur 5 surfaces et un F3P est quasi symetrique : la force
  laterale peut venir de `beta` autant que de l'inclinaison. Confirme que mu
  reste le degre de liberte purement CHOREGRAPHIQUE.

## M23. Les 4 % restants : deux causes, aucune musicale
Caracterisation des instants fautifs (32 beats, 60 deg, v~3 m/s) :

| grandeur | instants OK | instants KO |
|---|---|---|
| vitesse | 3.0 m/s | 3.0 m/s |
| acceleration | 3.7 m/s2 | **11.0 m/s2** |
| poussee | 28.7 % | **0 %** |

1. **7/11 sont contre le PLAFOND** (z = 6.0 m). C'est le
   `np.clip(p[2], z_range)` de `path_from_beats` : il ecrase la position contre
   la limite au lieu d'empecher d'y arriver, ce qui plie la geometrie. Meme
   defaut que le `np.clip` de `generator.py` cote drone.
   -> corriger la DIRECTION en amont (`choose_turn`), pas la position en aval.
2. **Les autres sont aux EXTREMITES** (t = 0.9 et t = 18.3-18.7). Le minimum
   snap sans conditions aux limites laisse v et a libres aux deux bouts et en
   profite : poussee a 0 %, l'avion est en chute libre.
   -> imposer v et a aux extremites.

Aucune des deux ne touche a la regle musicale.

### Principe acte avec l'utilisateur
Les waypoints ne dependent QUE de la musique. On ne supprime ni ne deplace un
waypoint parce que le suivi echoue. En revanche, la PLAGE d'angles est un
parametre de la regle (la musique fournit la FORCE ; la convertir en degres
demande une echelle, comme la taille de l'arene) : 30-150 deg venait du
QUADROTOR, qui pivote sur place. 20-60 deg convient a un avion, et le lien
musical est intact — un beat fort donne toujours un virage plus ample.

## M24. Deux correctifs a la source — la chaine musicale devient VOLABLE

**1. Extremites du minimum snap.** `minsnap.solve()` impose v=0 aux deux bouts
(correct pour un quadrotor qui decolle en stationnaire ; un avion a v=0 tombe —
d'ou poussee 0%, a=11 m/s2 mesures en M23). Ajout de `cruise_velocities()` dans
`minsnap_v2.py` : norme constante = vitesse de croisiere partout, direction en
bissectrice. Chemin toujours issu de la musique seule, seule la CIBLE DE SUIVI
change de conditions aux limites.

**2. Plafond de l'arene — recherche fine au lieu du clip.** Le clip pliait la
geometrie. Diagnostic (essentiel, evite une fausse piste) : minsnap est un
solve GLOBAL (continuite jusqu'au jerk sur TOUS les segments a la fois) — un
SEUL point plie corrompt l'acceleration ailleurs dans le polynome, parfois plus
fort LOIN du point que pres de lui (mesure : |a|=20.1 loin vs 8.1 pres, sur les
memes points). Une arene "genereuse mais finie" ne resout donc RIEN : 13 %
d'infaisabilite que 6 points touchent le bord ou seulement 2.

Solution : `path_from_beats` cherche maintenant le plan de rotation sur 72
candidats (au lieu de 4) qui minimise le depassement de l'arene. Verifie
geometriquement (exist.py) : a CHAQUE etape il existe toujours un plan qui
respecte l'arene (0/31 cas impossibles sur la vraie musique) — 4 candidats
etaient juste trop grossiers, la contrainte n'etait jamais reellement
irrespectable. L'AMPLITUDE du virage (= la force du beat) est inchangee, seul
le plan varie : purement geometrique.

L'ARENE EST GARDEE (decision explicite) : la trajectoire doit rester dans un
volume defini, sortir signifie ne pas suivre la choregraphie voulue.

| | avant | apres |
|---|---|---|
| arene 1-6 m, waypoints colles au bord | 6/32 | **0/32** |
| verdict solveur (musique reelle) | 13-16 % infaisable | **VOLABLE (0%)** |
| vitesse | 0.00-4.88 m/s (chute aux bords) | 2.80-3.69 m/s |

**JALON C ATTEINT : la chaine musique -> waypoints -> trajectoire ideale est
100 % volable sur la vraie chanson, arene comprise.**
