# Smart Object Removal — surveillance de vitrines

Projet semestriel 4 « Visual Tracking & intelligent applications » (axe Object Tracking).

Une caméra fixe en hauteur filme une ou plusieurs **vitrines** (ou étagères) d'une boutique. L'application
mémorise les articles exposés dans chaque vitrine, les suit dans le temps et **déclenche une alerte quand
quelqu'un prend un article**, en indiquant **quelle vitrine, quel article et quel client**. Une personne qui
passe devant sans rien prendre, une main qui manipule un article ou un article déplacé dans sa vitrine ne
déclenchent pas d'alerte.

---

## 1. Ce que fait l'application

| Étape | Rôle |
|---|---|
| **Inventaire** | Pendant les 2 premières secondes (`warmup_seconds`), l'application mémorise les articles présents dans chaque vitrine |
| **Suivi** | Chaque article garde un identifiant stable (DeepSORT) ; les clients sont suivis par un second DeepSORT, séparé |
| **Décision** | À chaque image, chaque article de l'inventaire est dans l'un des états ci-dessous |
| **Alerte** | Bandeau à l'écran, ligne dans `outputs/events.csv`, capture dans `outputs/captures/`, clip de preuve dans `outputs/clips/` |

États d'un article :

| État | Signification | Chronomètre |
|---|---|---|
| `PRESENT` | L'article est vu dans sa vitrine | remis à zéro |
| `MASQUE` | Pas vu, mais une personne (main, bras) le recouvre ou le touche | arrêté |
| `ABSENT` | Pas vu et rien ne le cache (ou sorti de sa vitrine) | tourne |
| `RETIRE` | Absent plus de 3 s **et** l'emplacement a changé visuellement : **alerte** | — |

Événements du journal : `INVENTAIRE`, `NOUVEL_OBJET` (article posé et immobile 2 s), `DEPLACE` (article
reposé ailleurs dans la même vitrine), `RETIRE` (avec `vitrine`, `personne_id` et `clip`), `REMIS` (article
revenu à sa place).

Règles conservées de la présentation de cadrage : inventaire pendant les premières secondes ; absent de sa
zone plus de 3 s sans être caché → `RETIRE` ; caché ou touché par une personne → `MASQUE`, chronomètre
arrêté ; déplacé dans la vitrine → `DEPLACE`, pas `RETIRE` ; vérification visuelle de l'emplacement avant
l'alerte ; événements `REMIS` et `NOUVEL_OBJET` ; journal CSV, captures, `tools/evaluate.py`.

## 2. Choix techniques

Le projet reprend l'architecture et les choix du dépôt
[kaustuv123/Missing-Object-Detection](https://github.com/kaustuv123/Missing-Object-Detection)
(YOLOv8n + DeepSORT + mémoire de référence de la scène, modules `detectors/`, `trackers/`, `engine/`,
`alerts/`, `utils/`) et y ajoute la logique de notre cadrage et le scénario boutique.

| Brique | Choix | Pourquoi |
|---|---|---|
| Détection (`detector.type: coco`) | **YOLOv8n** (Ultralytics 8.4.218), préentraîné COCO | Léger, temps réel sur CPU ; choix du dépôt de référence |
| Détection (`detector.type: world`) | **YOLO-World v2 s** (`yolov8s-worldv2.pt`) ou **YOLOE 26n** (`yoloe-26n-seg.pt`), vocabulaire ouvert | Articles de boutique hors COCO (chaussure, guitare, parfum…) décrits en texte dans `config.yaml`, sans fine-tuning |
| Détection (`detector.type: simulated`) | Détecteur par couleur (`detectors/color_detector.py`) | Uniquement pour la vidéo synthétique et les tests |
| Suivi des articles | **DeepSORT** (`deep-sort-realtime`) avec ReID MobileNetV2 | Kalman + IoU + **apparence** : un article caché par une main retrouve son identifiant |
| Suivi des clients | Second DeepSORT, séparé | Identifiant « client N » et trajectoire ; association à l'article retiré |
| Mémoire de scène | Inventaire de référence par vitrine (`engine/scene_monitor.py`) | Savoir ce qui *devrait* être là pour détecter ce qui manque |
| Vitrines | Polygones nommés dessinés au démarrage (`zones.json`, `<video>.zones.json`) | Configuration initiale autorisée par le cahier ; un article appartient à la vitrine qui contient le centre de sa boîte |
| Preuve | Tampon circulaire + clip 5 s avant / 5 s après chaque `RETIRE` | Vérification humaine de l'alerte |

Les deux modèles à vocabulaire ouvert ont été testés avec le package Ultralytics installé (8.4.218) :
`YOLOWorld(...).set_classes(liste)` et `YOLOE(...).set_classes(liste, modele.get_text_pe(liste))`,
avec téléchargement automatique du codeur de texte (CLIP, puis MobileCLIP). Les trois modèles testés
(`yolov8s-worldv2`, `yoloe-26n-seg`, `yoloe-11s-seg`) tournent entre 5 et 5,5 img/s sur CPU à `imgsz: 640`,
détection seule.

### Ce qui change par rapport au dépôt de référence

| Dépôt kaustuv123 | Ce projet |
|---|---|
| « DeepSORT » écrit à la main : Kalman + IoU, **sans descripteur d'apparence** (c'est en fait un SORT) | Vrai DeepSORT avec ReID MobileNetV2 (vignettes 128 px pour le CPU), ou histogramme de couleurs au choix |
| Un seul tracker, toutes classes confondues | Deux trackers séparés : articles et clients |
| Disparition = 15 images sans détection (0,5 s) : une main qui passe déclenche une alerte | Délai en **secondes** de vidéo (3 s), chronomètre arrêté quand une personne cache l'article |
| Toute l'image est surveillée | **Vitrines nommées** dessinées par l'utilisateur ; sortir de sa vitrine = retrait ; alerte avec le nom de la vitrine |
| Un nouvel identifiant après occultation = objet disparu + nouvel objet | Ré-association par position et par classe/vitrine : le changement d'identifiant est récupéré |
| Pas de vérification avant l'alerte | **Vérification visuelle** : si l'emplacement ressemble encore à l'article, c'est le détecteur qui l'a raté |
| Pas de déplacement, pas de retour | Événements `DEPLACE` et `REMIS` |
| Personne n'est associé à la disparition | **Client associé** (`personne_id`) : celui qui recouvre l'article ou le plus proche quand il disparaît ; trajectoire dessinée sur la capture |
| Classes COCO seulement (`classes: null`, tout ce que voit YOLOv8n) | COCO **ou** vocabulaire ouvert (YOLO-World / YOLOE) avec des articles de boutique en texte |
| Image entière à 540×900 | `detector.imgsz` réglable (640 / 1280) et `roi_crops` : recadrage agrandi de chaque vitrine pour les petits articles |
| Alertes à l'écran seulement, temps réel (horloge) | Journal CSV, captures, clips de preuve, résumés JSON, script d'évaluation ; temps de la vidéo, résultats reproductibles |
| Le moteur relit `config.yaml` et ignore `--video` | Ligne de commande complète (`--video`, `--detector`, `--select-zone`, `--no-show`, `--save-video`, `--save-clips`, `--out`…) |

Le modèle COCO reste interchangeable : `python main.py --model yolo26n.pt`.

## 3. Architecture

```
smart-object-removal/
├── main.py                      # point d'entrée (ligne de commande)
├── config.yaml                  # tous les paramètres
├── config_grand_magasin.yaml    # variante : rayon de grand magasin rempli de petits produits (YOLOE)
├── config_vetements.yaml        # variante : magasin de vêtements (vocabulaire vêtements / sacs, seuil 0,25)
├── detectors/
│   ├── base.py                  # socle : recadrage par vitrine (roi_crops), imgsz, dédoublonnage
│   ├── yolo_detector.py         # type coco : YOLOv8n (classes COCO) + personnes
│   ├── world_detector.py        # type world : YOLO-World / YOLOE, classes en texte + personnes
│   └── color_detector.py        # type simulated : couleur (vidéo synthétique et tests seulement)
├── trackers/deepsort_tracker.py # DeepSORT (articles) ; instancié une seconde fois pour les clients
├── engine/
│   ├── scene_monitor.py         # NOTRE LOGIQUE : inventaire par vitrine, états, événements, client associé
│   ├── zone.py                  # vitrines nommées (dessin, fichiers, compatibilité ancien zone.json)
│   ├── visual_check.py          # comparaison de l'emplacement avant l'alerte
│   ├── clip_buffer.py           # tampon circulaire et clips de preuve
│   └── pipeline.py              # boucle vidéo -> détection -> suivis -> moniteur -> alertes -> clips
├── alerts/notifier.py           # console, bandeaux, journal CSV (vitrine, personne_id, clip), captures
├── utils/                       # dessin (clients, trajectoires, panneau par vitrine), géométrie, FPS, types
├── tools/
│   ├── evaluate.py              # précision, rappel, F1, délai, fausses alertes (avec ou sans colonne vitrine)
│   ├── make_shop_video.py       # vidéo synthétique + zones + vérité terrain
│   ├── contact_sheet.py         # planche contact pour annoter
│   ├── check_camera.py          # la caméra est-elle fixe ?
│   ├── run_reference.py         # lance le code du dépôt de référence sur une vidéo
│   └── make_demo_video.py       # monte la vidéo démonstrative (cartons + vidéos annotées) -> outputs/demo_finale.mp4
├── tests/                       # tests sans caméra ni modèle
├── data/                        # SOURCES.md, annotations.csv, annotations_synth.csv
├── demo/                        # rendu : vidéo démonstrative, captures, journal et évaluation (versionnés)
└── videos/                      # vidéos de travail (non versionnées)
```

```
image -> détecteur (coco | world | simulated) -> articles + personnes
      -> DeepSORT articles (ID)        -> vitrine de chaque article
      -> DeepSORT clients (client N)   -> SceneMonitor (PRESENT / MASQUE / ABSENT / RETIRE, client associé)
      -> événements -> journal CSV + captures + clips + affichage
```

## 4. Installation

Python 3.9 à 3.13 (le projet a été développé et testé avec Python 3.13 sur Windows, CPU seulement).

```bash
python -m venv venv
venv\Scripts\activate             # Linux/macOS : source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # version CPU
pip install -r requirements.txt
```

Au premier lancement, les poids (`yolov8n.pt`, ou `yolov8s-worldv2.pt` + CLIP pour le mode `world`) sont
téléchargés automatiquement. `imageio-ffmpeg` fournit un binaire ffmpeg pour convertir les vidéos.

## 5. Utilisation

```bash
# 1. Dessiner les vitrines sur la première image (plusieurs polygones, un nom chacun)
#    -> enregistré dans videos/<video>.zones.json (prioritaire sur zones.json)
python main.py --video videos/web_mixkit_01_retrait.mp4 --select-zone

# 2. Lancer sur une vidéo (YOLOv8n COCO par défaut)
python main.py --video videos/web_mixkit_01_retrait.mp4

# 3. Articles de boutique hors COCO : vocabulaire ouvert
python main.py --video videos/web_mixkit_01_retrait.mp4 --detector world

# 4. Tout le dossier, sans fenêtre, vidéos annotées et clips
python main.py --video videos/ --no-show --save-video --save-clips --detector world

# 5. Vidéo synthétique (détecteur par couleur)
python tools/make_shop_video.py
python main.py --video videos/synth_boutique.mp4 --detector simulated

# Webcam
python main.py --video 0
```

Sélection des vitrines : clic gauche pour ajouter un point, clic droit pour retirer le dernier, `Entrée` pour
valider le polygone (le nom est demandé dans la console), `R` pour recommencer, `Entrée` sans point ou
`Échap` pour terminer. Sans vitrine, toute l'image est surveillée sous le nom « Zone 1 ». L'ancien
`zone.json` (une seule zone) est toujours lu.

Touches pendant la lecture : `q` quitter, `p` pause, `s` capture d'écran, `r` réapprendre l'inventaire.

Résultats dans `outputs/` :
- `events.csv` : un événement par ligne (`video, temps_s, image, evenement, vitrine, objet, classe,
  track_id, personne_id, x1..y2, disparu_a_s, personne_proche, capture, clip, details`) ;
- `captures/` : image annotée à chaque `RETIRE` et `REMIS`, avec la trajectoire du client associé ;
- `clips/` : clip annoté de 5 s avant à 5 s après chaque `RETIRE` ;
- `resumes/<video>.json` : durée, img/s, événements, vitrines, clients suivis, inventaire final ;
- `videos/` : vidéos annotées (avec `--save-video`).

### 5.1 Démonstration devant le jury (10 minutes)

Tout se lance depuis la racine du projet, environnement activé (`venv\Scripts\activate`). Chaque vidéo
ouvre la fenêtre « Smart Object Removal » (à gauche la scène annotée, à droite l'inventaire des vitrines et les
derniers événements) ; le chargement du modèle prend 10 à 15 s avant la première image.

```bash
# 1. Scénario complet maîtrisé (vidéo synthétique, 40 s) : 3 retraits attribués au bon client, 1 déplacement,
#    1 remise, un passant qui ne déclenche rien
python main.py --video videos/synth_boutique.mp4 --detector simulated

# 2. Vraie vitrine, vrai client : chaussure prise -> alerte RETIRE "client 2", trajectoire, clip de preuve
python main.py --video videos/web_mixkit_01_retrait.mp4 --detector world

# 3. Objet simple pris puis reposé : RETIRE à 5,1 s puis REMIS
python main.py --video videos/web_mixkit_06_retrait.mp4 --detector world

# 4. Grand magasin, rayon de 54 produits suivis un par un, cliente qui passe : aucune fausse alerte
python main.py --config config_grand_magasin.yaml --video videos/web_pexels_01_passage.mp4

# 5. Évaluation chiffrée et tests
python tools/evaluate.py --annotations data/annotations.csv
python -m pytest -q
```

À montrer ensuite : `outputs/events.csv` (journal), `outputs/captures/` (une image par alerte), `outputs/clips/`
(5 s avant / 5 s après), `outputs/evaluation.csv`, puis la section 7 de ce README (résultats et limites).
Pendant la lecture : `p` pause, `s` capture, `q` quitter. Si la machine est lente, ajouter `--no-show
--save-video` et projeter la vidéo annotée produite dans `outputs/videos/` avec VLC.

Vidéo démonstrative (rendu final) : `python tools/make_demo_video.py` enchaîne des cartons titres et les vidéos
annotées des scénarios 1 à 4, puis les trois cas limites caméra à la main, dans `outputs/demo_finale.mp4` (H.264, 2 min 54 s). Une copie est versionnée dans
`demo/` avec trois captures d'alerte, le journal `events.csv` et `evaluation_retire.csv`.

## 6. Vidéos de travail et évaluation

Nous n'avons pas encore de vidéo filmée par nous : les vidéos de travail sont des vidéos publiques, la vidéo
du dépôt de référence et une vidéo synthétique. Sources, licences, auteurs et contenu : **`data/SOURCES.md`**.

| Vidéo | Source | Caméra | Attendu |
|---|---|---|---|
| `web_mixkit_01_retrait.mp4` | Mixkit (Restricted License) | fixe | `RETIRE` ×2 (deux chaussures) |
| `web_mixkit_02_retrait.mp4` | Mixkit (Free License) | **mobile** (limite) | `RETIRE` (guitare) |
| `web_mixkit_03_manipulation.mp4` | Mixkit (Restricted License) | légèrement mobile | rien |
| `web_mixkit_04_retrait.mp4` | Mixkit (Restricted License) | fixe | `RETIRE` (robe décrochée du portant ; voir 7.6) |
| `web_mixkit_05_ajout.mp4` | Mixkit (Restricted License) | fixe | `NOUVEL_OBJET` ×2 (chaussures, sac posés ; voir 7.6) |
| `web_mixkit_06_retrait.mp4` | Mixkit (Restricted License) | quasi fixe (dérive 1,3 %) | `RETIRE` puis `REMIS` (verre pris sur un comptoir : cas le plus simple) |
| `web_pexels_02..04_retrait.mp4` | Pexels (Pexels License), choisies par l'étudiante | **à la main** (limites) | `RETIRE` (bouteille d'huile, portefeuille, sachet surgelé) : non détectés, voir 7.6 |
| `web_mendeley_01..03_retrait.mp4` | Shoplifting Dataset, CC BY 4.0 | fixe | `RETIRE` |
| `web_mendeley_04..06_passage.mp4` | Shoplifting Dataset, CC BY 4.0 | fixe | rien |
| `web_merl_01..02_passage.mp4` | MERL Shopping (recherche non commerciale) | fixe, plafond | rien |
| `web_pexels_01_passage.mp4` | Pexels (Pexels License) | fixe | rien (rayon de supermarché rempli, cliente qui passe ; voir 7.6) |
| `ref_kaustuv_bouteille.mp4` | dépôt de référence | téléphone à la main | `RETIRE` ×2 (bouteille) |
| `synth_boutique.mp4` | générée (`tools/make_shop_video.py`) | fixe | 3 `RETIRE`, 1 `DEPLACE`, 1 `REMIS`, clients B, D, F |

**Annotations** : `data/annotations.csv` (colonnes `video,vitrine,evenement,classe,temps_s` ; la colonne
`vitrine` et la classe sont facultatives, l'ancien format `video,evenement,classe,temps_s` est accepté) et
`data/annotations_synth.csv` (vérité terrain exacte, générée). Pour annoter une vidéo :
`python tools/contact_sheet.py videos/clip.mp4 --step 0.5` puis relever l'instant où l'article quitte sa place.

```bash
python main.py --video videos/ --no-show --save-video --detector world
python tools/evaluate.py --annotations data/annotations.csv
python tools/evaluate.py --annotations data/annotations_synth.csv --event DEPLACE
```

Le script affiche, par vidéo puis au total : vrais positifs, fausses alertes, retraits manqués, **précision,
rappel, F1, délai moyen d'alerte, fausses alertes par minute**, et les enregistre dans `outputs/evaluation.csv`.

## 7. Résultats obtenus (25 et 26 novembre 2026, CPU seul)

Tous les chiffres viennent de `outputs/evaluation*.csv`, `outputs_coco/evaluation.csv` et des résumés JSON ;
rien n'est arrondi à la main. Commandes : `python main.py --video videos/ --detector world --no-show --save-video`
(puis `--detector coco --out outputs_coco`, et `--roi-crops --imgsz 1280 --out outputs_roi` sur les vidéos Mendeley),
`python tools/evaluate.py`.

### 7.1 Vidéo synthétique (`detector.type: simulated`)

| Vérité terrain (`data/annotations_synth.csv`) | Obtenu (`outputs/events.csv`) | Délai |
|---|---|---|
| 12,5 s `RETIRE` parfum, Vitrine 2, client B | 15,67 s `RETIRE` parfum, Vitrine 2, client 1 | 3,2 s |
| 19,0 s `DEPLACE` montre, Vitrine 1 | 20,00 s `DEPLACE` montre, Vitrine 1 (58 px) | 1,0 s |
| 23,5 s `RETIRE` chaussure, Vitrine 1, client D | 26,67 s `RETIRE` chaussure, Vitrine 1, client 7 | 3,2 s |
| 29,5 s `REMIS` chaussure, Vitrine 1 | 30,40 s `REMIS` chaussure, Vitrine 1 | 0,9 s |
| 34,5 s `RETIRE` lunettes, Vitrine 2, client F (deux clients présents) | 37,63 s `RETIRE` lunettes, Vitrine 2, client 10 | 3,1 s |

`RETIRE` : précision 1,00, rappel 1,00, F1 1,00, délai moyen 3,16 s (le délai réglé est 3 s), 0 fausse alerte.
`DEPLACE` et `REMIS` : précision et rappel 1,00. Le client passant devant la vitrine 1 (bras devant un article
1 s) et le second client de la scène finale n'ont provoqué aucune alerte. Le retrait est attribué au bon client
dans les trois cas (les identifiants 1, 7, 10 sont ceux du tracker des clients).

### 7.2 Vidéos réelles, mode `world` (YOLO-World v2 s), événement `RETIRE`

| Vidéo | Réels | VP | FP | FN | Délai moyen | Commentaire |
|---|---|---|---|---|---|---|
| `web_mixkit_01_retrait` (vitrine chaussures) | 2 | 1 | 0 | 1 | 3,3 s | Les deux chaussures sont inventoriées à 2 s. La chaussure #1 passe en `MASQUE` pendant que l'homme l'examine devant la vitrine (8 à 16 s), puis `RETIRE` à 18,9 s, **client 2**, trajectoire dessinée, clip de preuve. La chaussure #2 est `ABSENT` (chronomètre lancé à 18,8 s) quand la vidéo s'arrête à 19,9 s : alerte attendue vers 21,8 s, hors vidéo. |
| `web_mixkit_02_retrait` (guitares, caméra mobile) | 1 | 1 | 0 | 0 | 1,8 s | **Vrai positif de coïncidence** : 7 guitares inventoriées ; le travelling fait sortir la guitare #1 du cadre, d'où le `RETIRE` à 10,8 s ; la guitare réellement décrochée (#5) n'est `ABSENT` que depuis 0,2 s à la fin. 5 `DEPLACE` parasites dus au mouvement de caméra. Cas limite assumé (voir `data/SOURCES.md`). |
| `web_mixkit_03_manipulation` (portant) | 0 | 0 | 0 | 0 | | Aucun article du vocabulaire détecté, aucune alerte. |
| `web_mixkit_04_retrait` (portants de robes, `config_vetements.yaml`, 26 nov.) | 1 | 0 | 0 | 1 | | Robe décrochée du portant : les vêtements sur cintres ne sont pas détectés individuellement (voir 7.6). |
| `web_mixkit_06_retrait` (verre sur un comptoir, 26 nov.) | 1 | 1 | 0 | 0 | 2,6 s | `RETIRE` à 5,13 s puis `REMIS` à 9,2 s (voir 7.6). |
| `web_pexels_02..04_retrait` (3 vidéos caméra à la main, 26 nov.) | 3 | 0 | 0 | 3 | | Articles trop petits ou tenus en main, caméra mobile : aucun inventaire exploitable (voir 7.6). |
| `web_mendeley_01..03_retrait` (table, petits objets) | 3 | 0 | 0 | 3 | | Le modèle voit le carton et la bouteille en inox (`box`, `bottle`) mais jamais le petit objet empoché. |
| `web_mendeley_04..06_passage` | 0 | 0 | 0 | 0 | | Aucune fausse alerte. |
| `web_merl_01..02_passage` (rayons vus du plafond, 2 min) | 0 | 0 | 0 | 0 | | Aucun produit du rayon détecté (emballages serrés, hors vocabulaire), aucune fausse alerte. |
| `ref_kaustuv_bouteille` (vidéo du dépôt de référence) | 2 | 0 | 0 | 2 | | La bouteille est détectée mais **tenue à la main dans la boîte de la personne** : elle n'est jamais « exposée », donc jamais inventoriée (voir 7.5). |
| **Total** (19 vidéos, dont 7 ajoutées le 26 nov.) | 13 | 3 | 0 | 10 | 2,58 s | **Précision 1,00, rappel 0,23, F1 0,38, 0,00 fausse alerte / min** sur 6 min de vidéo (`outputs/evaluation.csv`) |

Captures : `outputs/captures/web_mixkit_01_retrait_000566_RETIRE_obj1.jpg` (chaussure, client 2 et sa
trajectoire) et `outputs/captures/web_mixkit_02_retrait_000325_RETIRE_obj1.jpg` ; clips correspondants dans
`outputs/clips/`.

Analyse des retraits manqués : 6 sur 10 viennent de la **détection** (objets Mendeley trop petits et sans nom
dans le vocabulaire, robes sur cintres non séparables, bouteilles et portefeuilles trop petits ou trop loin
dans les vidéos Pexels à la main), 3 de la **règle** « un article tenu par une personne n'est pas un article
exposé » (vidéo de référence, sachet surgelé tenu du début à la fin), 1 de la **vidéo** (clip Mixkit trop court).
Aucun ne vient du suivi ni des vitrines. Réglages proposés : pour Mendeley, ajouter au vocabulaire les mots
qui décrivent les objets réellement posés (`packet`, `small box`, `phone`) et filmer plus près ; pour une
vidéo courte, abaisser `removal_delay_seconds` à 2 s (`--delay 2`).

### 7.3 Comparaison `coco` / `world` sur les vidéos réelles

Articles = classes inventoriées ou ajoutées (`INVENTAIRE`, `NOUVEL_OBJET`) ; img/s = chaîne complète
(détection + deux DeepSORT + affichage + vidéo annotée) sur CPU, images de 960 px, sans autre charge.
Tableau produit par `python tools/compare_runs.py outputs_coco outputs --names coco world`.
Mesures du 25 novembre (vocabulaire sans `cup`). Le 26 novembre, après l'ajout de `cup` au vocabulaire
(section 7.6), les 12 vidéos ont été retraitées : seuls les événements de `web_mendeley_04` changent (indiqué
dans le tableau), les autres lignes sont identiques.

| Vidéo | coco : articles | coco : événements | coco img/s | world : articles | world : événements | world img/s |
|---|---|---|---|---|---|---|
| ref_kaustuv_bouteille | — | — | 7,4 | — | — | 4,5 |
| web_mendeley_01_retrait | bottle×2, suitcase×2 | NOUVEL_OBJET 2 | 4,4 | bottle×2, box | NOUVEL_OBJET 1 | 3,5 |
| web_mendeley_02_retrait | bottle | — | 4,8 | bottle, box | — | 3,6 |
| web_mendeley_03_retrait | bottle, cell phone, cup | NOUVEL_OBJET 2 | 5,0 | box×2 | DEPLACE 2, NOUVEL_OBJET 1 | 3,5 |
| web_mendeley_04_passage | bottle | — | 5,6 | bottle, box×2 | NOUVEL_OBJET 1 (26 nov., avec `cup` : DEPLACE 1, NOUVEL_OBJET 2) | 2,6 |
| web_mendeley_05_passage | suitcase | — | 5,5 | box | — | 3,0 |
| web_mendeley_06_passage | suitcase×2 | NOUVEL_OBJET 2 | 5,6 | box | — | 3,4 |
| web_merl_01_passage | — | — | 8,4 | — | — | 3,2 |
| web_merl_02_passage | — | — | 8,5 | — | — | 3,4 |
| web_mixkit_01_retrait | cell phone | NOUVEL_OBJET 1 | 7,9 | **shoe×2** | **RETIRE 1** | 3,8 |
| web_mixkit_02_retrait | clock×2, vase | DEPLACE 3, NOUVEL_OBJET 2, RETIRE 1 | 5,9 | guitar×7 | DEPLACE 5, RETIRE 1 | 2,4 |
| web_mixkit_03_manipulation | — | — | 8,6 | — | — | 4,7 |

Lecture : YOLOv8n COCO ne connaît ni les chaussures ni les guitares (il y voit `cell phone`, `clock`, `vase`)
et obtient précision 1,00 mais rappel 0,12 (son seul « vrai positif » est la fausse horloge de la vidéo
guitares). YOLO-World reconnaît les articles de boutique nommés en texte (`shoe`, `guitar`) au prix d'une
vitesse divisée par deux (5,5 img/s en détection seule à `imgsz 640`, 2,4 à 4,7 img/s pour la chaîne
complète). Seul le mode `coco` atteint l'objectif de 5 img/s du dépôt de référence ; en mode `world`, on traite
une vidéo de surveillance en différé ou en sautant des images.

### 7.4 Caméra en hauteur et petits articles (`roi_crops` + `imgsz 1280`)

Essai sur `web_mendeley_01` et `_02` (mode `world`, dossier `outputs_roi/`) : 0,45 img/s, articles inventoriés
`perfume bottle` + `bottle` et `box` + `bottle`, toujours aucun `RETIRE` : l'agrandissement de la table ne
suffit pas à faire reconnaître le petit objet empoché. L'option reste utile quand l'article est petit **mais
nommé** dans le vocabulaire ; l'essai a été arrêté après deux vidéos (12 min de calcul chacune).

### 7.5 Comparaison avec le dépôt de référence sur sa vidéo (`ref_kaustuv_bouteille.mp4`)

Le code du dépôt est lancé tel quel par `python tools/run_reference.py` (après `git clone` du dépôt dans
`reference/Missing-Object-Detection`, dossier non versionné, supprimé du projet une fois la comparaison faite) (copie de sa config avec notre chemin de
vidéo, fenêtres neutralisées, messages console horodatés) ; ses alertes sont dans `outputs/reference_events.csv`.
Vérité : la bouteille tenue devant la caméra disparaît à 5,3 s et à 9,3 s ; la caméra bouge fortement après 11 s.

| Instant | Dépôt kaustuv123 (YOLOv8n + SORT, 15 images) | Ce projet (`coco` ou `world`, règle « article tenu » active ou non) |
|---|---|---|
| 0,2 s | « Baseline established with 1 object » (la personne) | Inventaire vide : la personne n'est pas un article |
| 3,1 s | `MISSING` objet 2 (détection fugace) : **fausse alerte** | — |
| 5,6 s | `MISSING` objet 3 : **bouteille disparue, juste** (délai 0,3 s) | — : la bouteille est dans la boîte de la personne (tenue), jamais « posée » 2 s immobile |
| 9,9 s | `MISSING` objet 4 : **bouteille disparue, juste** (délai 0,6 s) | — (même raison) |
| 12,0 s et 12,2 s | `MISSING` objets 1 et 5 : **fausses alertes** (la caméra bouge) | — |
| 3,5 à 9,3 s | 10 messages `REAPPEARED` (identifiants instables sans ReID) | — |

Le dépôt de référence réagit en 0,5 s à toute disparition d'une boîte, ce qui donne ici 2 alertes justes et
3 fausses ; notre chaîne, conçue pour des articles posés dans une vitrine, ne considère pas un objet tenu à la
main comme un article exposé et ne produit rien sur cette vidéo, ni alerte ni fausse alerte. Même avec la
règle désactivée (`candidate_person_coverage: 1.1`, dossiers `outputs_ref_norule_*`), la bouteille n'est
jamais immobile 2 s : aucune alerte en `coco` ; en `world`, seuls les lunettes et le t-shirt portés par la
personne entrent dans l'inventaire, sans alerte ensuite. Cette vidéo illustre une différence de cahier des charges plus qu'une
différence de qualité.

### 7.6 Scénarios ajoutés le 26 novembre : grand magasin, magasin de vêtements, objet simple, caméra à la main

**Grand magasin : rayon rempli de petits produits** (`config_grand_magasin.yaml`).

Vidéo `web_pexels_01_passage.mp4` (rayon produits laitiers d'un supermarché, caméra fixe, plusieurs
centaines de produits, une cliente longe le rayon de 7,7 à 13,4 s sans rien prendre ; vitrines `Rayon haut`
et `Rayon bas`). Avec `config.yaml` (YOLO-World, vocabulaire boutique), **aucun produit** n'est détecté,
même à `imgsz 1280` et seuil 0,1 : les mots `shoe`, `guitar`… ne décrivent rien de ce rayon et YOLO-World
voit mal les petits objets. D'où la variante `config_grand_magasin.yaml` :

| Réglage | `config.yaml` | `config_grand_magasin.yaml` | Pourquoi |
|---|---|---|---|
| modèle `world` | YOLO-World v2 s | **YOLOE 26n** | sur une image du rayon : YOLO-World 0 à 2 produits, YOLOE 49 (640 px) à 150 (960 px) |
| `world_classes` | shoe, guitar… | **product**, bottle, yogurt, milk carton, cheese | vocabulaire générique |
| `confidence` / `imgsz` | 0,35 / 640 | **0,15 / 960** | à 0,35 seuls 0 à 6 produits passent ; à 0,15 et 960 px : 55 à 75 |
| `tracker.embedder` | mobilenet | **histogram** | 60 produits par image : le ReID MobileNetV2 fait tomber la chaîne à 0,65 img/s |
| `tracker.max_age` / `max_iou_distance` | 90 / 0,7 | **30 / 0,5** | moins d'échanges d'identifiants entre produits identiques voisins |
| `monitor.move_ratio` | 0,5 | **1,5** | la boîte d'un produit de 40 px tremble de plus d'une demi-diagonale |
| `monitor.rebind_max_distance_ratio` | null | **3,0** | un article manquant n'est « retrouvé » que dans un objet posé à moins de 3 diagonales (nouveau paramètre, test `test_limite_de_distance_pour_reconnaitre_un_article_deplace`) |

```bash
python main.py --config config_grand_magasin.yaml --video videos/web_pexels_01_passage.mp4 --save-video
```

| Passage | Inventaire à 2 s | `RETIRE` | `DEPLACE` | `NOUVEL_OBJET` | img/s |
|---|---|---|---|---|---|
| `config.yaml` (YOLO-World) | 0 | 0 | 0 | 0 | 4,5 |
| variante, avant réglage du suivi (ReID MobileNet, `move_ratio` 0,5, sans limite de ré-association) | 54 | 0 | 41 | 22 | 0,65 |
| **variante finale** | **54** (48 rayon bas, 6 rayon haut) | **0** | 11 | 24 | **3,1** |

Résultat attendu sur cette vidéo : aucune alerte. Obtenu : **aucune fausse alerte `RETIRE`** (précision 1,00
conservée dans `outputs/evaluation.csv`), 54 produits suivis individuellement, les produits derrière la cliente
passent en `MASQUE (client 3)` puis reviennent `PRESENT`. Les 11 `DEPLACE` restants (100 à 125 px) et les
24 `NOUVEL_OBJET` sont des produits identiques voisins dont les détections s'échangent pendant le passage de
la cliente : 23 produits sont encore `ABSENT` à la fin (ils seront `RETIRE` après 3 s si la vidéo continuait
sans qu'ils soient revus). Sur un rayon dense, la limite est donc le suivi de produits identiques, pas la
détection ni la règle d'alerte. Sans vidéo de ce type où un produit est réellement pris, le rappel n'est pas
mesurable ici (voir la section 10 pour filmer ce cas).

**Magasin de vêtements** (`config_vetements.yaml` : mode `world`, vocabulaire `dress, shirt, jacket, sweater,
clothes, handbag, purse, shoe, hat`, seuil 0,25 ; deux vidéos Mixkit à caméra fixe ajoutées le 26 novembre) :

```bash
python main.py --config config_vetements.yaml --video videos/web_mixkit_04_retrait.mp4 --save-video
python main.py --config config_vetements.yaml --video videos/web_mixkit_05_ajout.mp4 --save-video
```

| Vidéo | Attendu | Obtenu | Lecture |
|---|---|---|---|
| `web_mixkit_04_retrait` (deux portants de robes, étagère avec un sac ; la cliente blonde décroche une robe blanche à 16 s) | `RETIRE` robe, `Portant robes` | `NOUVEL_OBJET` **purse** à 9,9 s (le sac de l'étagère, vu avec 0,16 à 0,56 de confiance selon l'image), puis `MASQUE (client 2)` quand la cliente passe devant ; **aucun `RETIRE`** (3,7 img/s) | Les robes sur cintres ne sont **pas séparables** par un détecteur à vocabulaire ouvert : YOLO-World et YOLOE voient le portant entier comme une seule boîte `clothes` à 0,15 - 0,24 de confiance (même à `imgsz 1280`, même sur un recadrage agrandi, avec les mots `dress`, `hanging dress`, `garment`, `blouse`). Retrait manqué (faux négatif), par la **détection**, pas par les règles. |
| `web_mixkit_05_ajout` (présentoir : une main pose des chaussures dorées 3 - 6 s puis un sac 7 - 10 s) | `NOUVEL_OBJET` chaussures, `NOUVEL_OBJET` sac | `INVENTAIRE` **clothes** (la robe du mannequin) à 2 s, `NOUVEL_OBJET` **shoe** à 12,7 s (3,1 img/s) | Chaussures reconnues ; l'article n'est compté « posé » que 2 s après le départ de la main (10,5 s), d'où le délai. Le sac n'est pas séparé des chaussures (même boîte `shoe`) : faux négatif. Avant la règle anti-doublon ci-dessous, la robe du mannequin produisait 3 `NOUVEL_OBJET` de plus (`shirt`, `dress`, `sweater` : même boîte, autre nom). |

Nouveau paramètre `monitor.duplicate_iou` (0,7) : un objet qui recouvre à plus de 70 % un article déjà
inventorié n'est pas un nouvel article, quel que soit son nom (test
`test_meme_article_sous_un_autre_nom_n_est_pas_un_nouvel_objet`). Vidéo annotée :
`outputs/videos/web_mixkit_05_ajout_annotee.mp4` (les captures ne sont prises que pour `RETIRE` et `REMIS`).
Évaluation `--event NOUVEL_OBJET` sur toutes les vidéos : 1 vrai positif (chaussures, délai 9,2 s), 1 manqué (sac),
24 faux positifs tous sur le rayon de supermarché (produits identiques, voir ci-dessus) ; précision 0,03.

**Objet simple pris puis reposé** (`web_mixkit_06_retrait.mp4`, verre sur un comptoir de bar, mot `cup` ajouté au
vocabulaire de `config.yaml` le 26 novembre) :

```bash
python main.py --video videos/web_mixkit_06_retrait.mp4 --detector world
```

| Attendu | Obtenu |
|---|---|
| `RETIRE` verre à 2,5 s, `REMIS` à 7,5 s | `INVENTAIRE cup` à 2 s, **`RETIRE` à 5,13 s** (3,0 s d'absence, similarité visuelle 0,00), **`REMIS` à 9,2 s** |

C'est le cas le plus simple et le plus lisible : objet rigide, isolé, contrasté, caméra quasi fixe. Leçon tirée :
avec `glass` **et** `cup` dans le vocabulaire, le même verre était nommé tantôt l'un, tantôt l'autre, la classe
de sa piste changeait d'une image à l'autre et il n'était jamais inventorié (section 8, « pas de synonymes »).
Pas de « client N » ici : seule une main entre dans le champ, et une main seule n'est pas une personne pour
le détecteur.

**Trois vidéos Pexels choisies par l'étudiante, caméra à la main** (`web_pexels_02..04`, sources section 4.2 de
`data/SOURCES.md`) :

| Vidéo | Config | Attendu | Obtenu | Pourquoi |
|---|---|---|---|---|
| `web_pexels_02_retrait` (homme qui prend une bouteille d'huile dans un rayon, caméra qui le suit) | `config.yaml`, `world` | `RETIRE` bottle | aucun inventaire, aucun événement (3,8 img/s) | les bouteilles du rayon sont trop petites et trop loin au début (aucune détection `bottle` avant qu'il en tienne une à la main, donc jamais « exposée ») ; la caméra bouge |
| `web_pexels_03_retrait` (maroquinerie, portefeuille pris sur l'étagère puis reposé, caméra qui bascule) | `config_vetements.yaml` (+ `wallet`) | `RETIRE` puis `REMIS` wallet | `NOUVEL_OBJET handbag` à 9,3 s (le sac de gauche, vu à 0,17 - 0,23 puis plus), rien d'autre (4,6 img/s) | les portefeuilles alignés sur l'étagère ne sont détectés que **tenus en main** (`wallet` 0,39 - 0,44) ; l'étagère glisse de 80 px avec la caméra |
| `web_pexels_04_retrait` (sachet surgelé pris dans un congélateur, gros plan à la main) | `config_grand_magasin.yaml` | `RETIRE` | 4 `NOUVEL_OBJET product`, aucun `RETIRE` (4,3 img/s) | caméra très mobile ; le sachet est tenu par la main du début à la fin, jamais posé |

Ces trois vidéos sortent du cadre du projet (caméra fixe en hauteur) et sont gardées comme **cas limites**
documentés : aucune fausse alerte, mais aucun retrait détecté. Elles montrent ce qu'il faut pour que la chaîne
fonctionne : caméra fixe, articles visibles et séparés dès le début, un article « pris » qui quitte le champ.


### 7.7 Limites connues

- **Détection** : les articles non reconnus (petits objets Mendeley, emballages MERL, vêtements sur cintres vus
  comme un seul bloc) ne peuvent pas être surveillés ; `world` dépend des mots choisis dans `world_classes`. Pas de fine-tuning dans ce projet.
- **Caméra mobile** : vérification visuelle, vitrines fixes et `DEPLACE` ne sont plus valables (vidéo guitares).
- **Une personne qui reste devant la vitrine** maintient les articles en `MASQUE` : l'alerte n'arrive qu'après
  son départ (chaussures : 2,6 s après) ou après `max_occlusion_seconds`.
- **Articles identiques côte à côte** (cartons Mendeley, guitares, rayon de supermarché) : les détections
  sautent d'un article à l'autre et produisent des `DEPLACE` et `NOUVEL_OBJET` parasites ; sur un rayon
  dense, `config_grand_magasin.yaml` les réduit (41 → 11 `DEPLACE`) sans les supprimer.
- **Vitesse** : 2,4 à 4,7 img/s en mode `world` sur ce CPU (8 cœurs) ; en direct, il faut sauter des images.
- **Vidéos courtes** : une alerte a besoin de 3 s d'absence visible après le départ de la personne.
- Les instants annotés dans `data/annotations.csv` ont été relevés par lecture de planches contact (0,5 s) ;
  ils peuvent être affinés.


## 8. Paramètres principaux (`config.yaml`)

| Paramètre | Défaut | Effet |
|---|---|---|
| `detector.type` | `coco` | `coco` (YOLOv8n COCO), `world` (vocabulaire ouvert), `simulated` (couleur) |
| `detector.target_classes` | bouteille, sac à main… | articles surveillés en mode `coco` (noms COCO) |
| `detector.world_model` | `yolov8s-worldv2.pt` | modèle `world` ; `yoloe-26n-seg.pt` pour YOLOE |
| `detector.world_classes` | shoe, guitar, handbag, cup… | articles en texte libre en mode `world` ; **pas de synonymes** (`glass` + `cup`) : le même objet serait nommé tantôt l'un, tantôt l'autre, la classe de sa piste changerait et il ne serait jamais inventorié (constaté sur `web_mixkit_06`) |
| `detector.confidence` | 0.35 | seuil de confiance |
| `detector.imgsz` | 640 | taille d'inférence (1280 pour les petits articles, 4 × plus lent) |
| `detector.roi_crops` | false | articles cherchés dans un recadrage agrandi de chaque vitrine (personnes : image entière) |
| `tracker.embedder_size` | 128 | côté des vignettes du ReID (224 dans la bibliothèque, 3 × plus lent) |
| `person_tracker.enabled` | true | second DeepSORT pour les clients |
| `zone.file` | `zones.json` | vitrines ; `<video>.zones.json` est prioritaire ; l'ancien `zone.json` est lu |
| `monitor.removal_delay_seconds` | 3.0 | absence non cachée avant l'alerte |
| `monitor.max_occlusion_seconds` | 15.0 | au-delà, un article caché compte comme absent |
| `monitor.occluder_margin` | 0.5 | une personne à moins de 0,5 diagonale « cache » l'article |
| `monitor.visual_check` / `visual_similarity` | true / 0.75 | vérification visuelle avant l'alerte |
| `monitor.person_max_distance_ratio` | 4.0 | distance maximale (en diagonales) pour associer un client |
| `monitor.duplicate_iou` | 0.7 | un objet recouvrant à plus de 70 % un article inventorié n'est pas un nouvel article, même sous un autre nom |
| `monitor.rebind_max_distance_ratio` | null | distance maximale (en diagonales) pour reconnaître un article manquant dans un objet qui vient de se poser ; `3.0` dans `config_grand_magasin.yaml` |
| `monitor.candidate_person_coverage` | 0.6 | un objet recouvert à plus de 60 % par une personne (lunettes portées, article en main) n'entre pas dans l'inventaire |
| `output.save_clips`, `clip_pre_seconds`, `clip_post_seconds` | true, 5, 5 | clips de preuve |

## 9. Tests

```bash
pytest -q
```

Les tests simulent les sorties des détecteurs et des trackers image par image, sans modèle : inventaire,
retrait après 3 s, main qui passe, article caché, occultation trop longue, changement d'identifiant,
déplacement (même loin de sa place, sous un nouvel identifiant), sortie de zone, nouvel objet, retour,
vérification visuelle ; **deux vitrines avec retrait dans une seule, association du bon client parmi deux,
article passé d'une vitrine à l'autre, compatibilité de l'ancien `zone.json`, lecture des annotations avec et
sans colonne vitrine**, détecteur par couleur, recadrage `roi_crops`, clips, journal CSV.

## 10. Protocole pour filmer notre propre vitrine simulée

La soutenance exige une vidéo inédite filmée par nous. Protocole :

1. **Vitrine** : une étagère ou une table contre un mur, fond uni ou peu chargé, avec **3 articles** bien
   séparés (au moins une largeur d'article entre eux) et de couleurs différentes du fond. Pour le mode `coco`,
   choisir des classes COCO (bouteille, tasse, livre, sac à main, téléphone) ; pour le mode `world`, n'importe
   quel article nommé dans `world_classes` (chaussure, montre, parfum…).
2. **Caméra** : téléphone **fixé** (trépied ou appui stable, jamais tenu à la main) **en hauteur dans un
   coin**, à 1,5–2,5 m de l'étagère, en plongée d'environ 30–45°, horizontal (paysage), 1080p à 30 img/s,
   exposition et mise au point verrouillées, lumière stable (pas de contre-jour).
3. **Début de clip** : **3 s immobiles sans personne** pour l'inventaire, puis l'action, puis 5 s après.
4. **Clips de 30 s**, un scénario par clip, chaque scénario au moins trois fois :
   - un client prend un article et sort du champ (attendu : `RETIRE`, bon client) ;
   - un client passe devant, touche ou soulève et repose l'article (attendu : rien, ou `DEPLACE`) ;
   - un client déplace l'article dans la vitrine (attendu : `DEPLACE`) ;
   - un client prend un article et le remet 10 s plus tard (attendu : `RETIRE` puis `REMIS`) ;
   - deux clients, un seul prend (attendu : `RETIRE` attribué au bon client) ;
   - test final filmé dans un autre lieu, gardé pour la fin.
5. **Nommage et annotation** : `videos/perso_<numero>_<scenario>.mp4`, conversion
   `ffmpeg -i in.mp4 -vf "scale=960:-2,fps=30" -c:v libx264 -preset veryfast out.mp4`,
   vitrines dessinées avec `--select-zone`, instants relevés avec `tools/contact_sheet.py` dans
   `data/annotations.csv`, caméra vérifiée avec `tools/check_camera.py`.

## Crédits

- Architecture et choix YOLOv8n + DeepSORT + mémoire de scène : [kaustuv123/Missing-Object-Detection](https://github.com/kaustuv123/Missing-Object-Detection) (vidéo de test « Whatsapp Vid 2.mp4 » utilisée pour la comparaison)
- Détection : [Ultralytics YOLO](https://docs.ultralytics.com/) (YOLOv8, YOLO-World, YOLOE)
- Suivi : [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime)
- Vidéos : Mixkit (m0de0n, rubenvelasco, FrameStock) ; Shoplifting Dataset (2022), Mohd. Aquib Ansari et Dushyant Kumar Singh, CV Laboratory MNNIT Allahabad, CC BY 4.0, DOI 10.17632/r3yjf35hzr.1 ; MERL Shopping Dataset, Mitsubishi Electric Research Labs (Singh, Marks, Jones, Tuzel, Shao, CVPR 2016). Détails : `data/SOURCES.md`.
