# Sources des vidéos de travail

Toutes les vidéos sont dans `videos/` (non versionné). Les originaux téléchargés ne sont pas conservés
(supprimés le 26/11/2026 pour libérer 1 Go ; les URL ci-dessous permettent de les retélécharger). Conversion : ffmpeg (binaire du paquet `imageio-ffmpeg` 0.8.1,
ffmpeg 9.0.2), MP4 H.264 `-preset veryfast -crf 23`, 30 img/s, 960 px de large
(sauf la vidéo du dépôt de référence, en portrait, gardée à 480 px de large).

Vérification de caméra fixe : `python tools/check_camera.py videos/*.mp4` (médiane des différences
de fond entre images consécutives, et translation globale par corrélation de phase). Une vidéo
est dite FIXE si la différence de fond est < 2,5 niveaux de gris et le décalage < 1 % de la largeur.

Date de téléchargement des sources : 25 novembre 2026 (section 4 : 26 novembre 2026).

## 1. Mixkit

Licences lues sur chaque page le 25/11/2026. Mixkit propose deux licences pour les vidéos :
« Mixkit Stock Video Free License » et « Mixkit Restricted License » (toutes deux gratuites,
sans attribution obligatoire ; la licence Restricted interdit la revente et l'usage isolé de la
vidéo). Le texte complet des licences est affiché par JavaScript sur https://mixkit.co/license/
et n'a pas pu être lu automatiquement : usage validé par le responsable du projet pour un
travail universitaire non redistribué.

| Fichier | Titre | URL | Auteur | Licence | Caméra | Contenu / usage |
|---|---|---|---|---|---|---|
| `web_mixkit_01_retrait.mp4` (19,9 s) | Pair of shoes on a sideboard are taken by a man | https://mixkit.co/free-stock-video/pair-of-shoes-on-a-sideboard-are-taken-by-a-35811/ | m0de0n | Mixkit Restricted License | **FIXE** (diff 0,89, décalage 0,4 px) | Deux mocassins exposés sur une console ; un homme prend le premier vers 8,3 s, l'examine devant la console, puis prend le second vers 15,6 s et s'en va. **Cas principal** : retraits à détecter, deux alertes attendues. |
| `web_mixkit_02_retrait.mp4` (11,1 s) | Young Musician Picking Electric Guitar in Store | https://mixkit.co/free-stock-video/young-musician-picking-electric-guitar-in-store-50303/ | rubenvelasco | Mixkit Stock Video Free License | **MOBILE** (diff 2,95, décalage 69 px : travelling latéral) | Un musicien de dos choisit une guitare sur un présentoir mural et la décroche vers 9 s. Gardé **seulement comme exemple de limite** (caméra mobile : la vérification visuelle et les zones fixes ne sont pas valables). |
| `web_mixkit_03_manipulation.mp4` (9,9 s) | Hand selecting through clothes | https://mixkit.co/free-stock-video/hand-selecting-through-clothes-23327/ | FrameStock | Mixkit Restricted License | **MOBILE** (diff 3,90, décalage 15 px : caméra à la main, léger) | Une main fait glisser des cintres sur un portant sans rien prendre. Test de fausses alertes (aucune alerte attendue) ; léger mouvement de caméra à garder en tête. |
| `web_mixkit_04_retrait.mp4` (21,6 s) | Two woman picking dresses in the clothing store | https://mixkit.co/free-stock-video/two-woman-picking-dresses-in-the-clothing-store-24410/ | (auteur affiché sur la page Mixkit) | Mixkit Restricted License (720p) | **FIXE** (diff 0,95, décalage 0,3 px) | Magasin de vêtements : portant de robes au centre, portant bleu à gauche, sac sur l'étagère en haut à gauche. Deux clientes regardent les robes ; la cliente blonde décroche une robe blanche du portant vers 16 s et la garde en main jusqu'à la fin. Téléchargée le 26/11/2026 ; vitrines `Portant robes`, `Portant gauche`, `Etagere sacs`. |
| `web_mixkit_05_ajout.mp4` (14,3 s) | Arranging a pair of elegant slippers and a dress | https://mixkit.co/free-stock-video/arranging-a-pair-of-elegant-slippers-and-a-dress-35877/ | (auteur affiché sur la page Mixkit) | Mixkit Restricted License (720p) | **FIXE** (diff 0,41, décalage 0,2 px) | Présentoir d'une boutique de mode : une main pose des chaussures dorées (3 à 6 s) puis un sac (7 à 10 s) devant une robe. Test de l'événement `NOUVEL_OBJET` (articles ajoutés, aucun retrait). Téléchargée le 26/11/2026 ; vitrine `Presentoir`. |
| `web_mixkit_06_retrait.mp4` (9,8 s) | A man's hand takes a glass from a bar counter | https://mixkit.co/free-stock-video/a-mans-hand-takes-a-glass-from-a-bar-counter-43970/ | (auteur affiché sur la page Mixkit) | Mixkit Restricted License (720p) | **quasi FIXE** (diff 0,69, décalage 4,2 px soit 1,3 % : légère dérive, au-dessus du seuil de 1 %) | Comptoir de bar : un verre posé ; une main le prend à 2,5 s, l'emplacement reste vide, le verre est reposé à 7,5 s. **Cas « article pris » le plus simple** : objet rigide, isolé, bien contrasté (`RETIRE` puis `REMIS` attendus). Téléchargée le 26/11/2026 ; vitrine `Comptoir`. |

Version téléchargée : HD 720p (1280×720), le 1080p n'étant pas disponible pour deux des trois vidéos.

## 2. Shoplifting Dataset (2022), CV Laboratory MNNIT Allahabad

- Auteurs : Mohd. Aquib Ansari et Dushyant Kumar Singh.
- URL : https://data.mendeley.com/datasets/r3yjf35hzr/1 — DOI 10.17632/r3yjf35hzr.1
- Licence : **CC BY 4.0** (indiquée sur la page du jeu de données).
- Citation : Ansari, Mohd. Aquib; Singh, Dushyant Kumar (2023), « Shoplifting Dataset (2022) - CV Laboratory
  MNNIT Allahabad », Mendeley Data, V1, doi: 10.17632/r3yjf35hzr.1
- Archive `Dataset.zip` (760 Mo) : 93 vidéos « Shoplifting » et 91 « Normal », 640×480, 30 img/s,
  10 à 12 s chacune, scènes jouées dans un laboratoire : une table avec des cartons, une bouteille
  et de petits objets, caméra fixe en hauteur.
- Caméra : **FIXE** (diff 0,31 à 0,49, décalage ≤ 1,4 px).

| Fichier | Original | Contenu / usage |
|---|---|---|
| `web_mendeley_01_retrait.mp4` | `Shoplifting (1).mp4` | Un homme prend un petit objet sur la table vers 2,3 s et le met dans sa poche. Retrait attendu. |
| `web_mendeley_02_retrait.mp4` | `Shoplifting (2).mp4` | Un homme casquette arrive, prend un petit objet blanc vers 7,0 s, l'empoche. Retrait attendu. |
| `web_mendeley_03_retrait.mp4` | `Shoplifting (3).mp4` | Un homme manipule un objet, le soulève vers 4,5 s, le porte à son visage puis part avec. Retrait attendu. |
| `web_mendeley_04_passage.mp4` | `Normal (2).mp4` | Un homme casquette regarde les objets sur la table sans rien emporter. Aucune alerte attendue. |
| `web_mendeley_05_passage.mp4` | `Normal (3).mp4` | Un homme en chemise à carreaux examine les objets de la table. Aucune alerte attendue. |
| `web_mendeley_06_passage.mp4` | `Normal (4).mp4` | Un homme tient un objet orange, le regarde et le repose. Aucune alerte attendue. |

Limite : les objets de la table sont petits et ne sont pas des classes COCO (sauf la bouteille) ;
ces vidéos servent surtout au mode `world` et au test des fausses alertes.

## 3. MERL Shopping Dataset (Mitsubishi Electric Research Labs)

- URL : https://merl.com/research/downloads/MERL_Shopping_Dataset (données :
  https://www.merl.com/pub/tmarks/MERL_Shopping_Dataset/, copie : https://huggingface.co/datasets/Voxel51/MERL_Shopping_Dataset)
- Auteurs : B. Singh, T. K. Marks, M. J. Jones, O. Tuzel, M. Shao, « A Multi-Stream Bi-Directional
  Recurrent Neural Network for Fine-Grained Action Detection », CVPR 2016.
- Licence : la page MERL n'affiche pas de licence ; la fiche Hugging Face indique
  **« Non-commercial research use only »** (copyright MERL, voir `TR2016-080.pdf`). Usage validé par le
  responsable du projet pour un travail universitaire.
- 106 vidéos de 2 min, 920×680, 30 img/s, caméra fixe au plafond au-dessus de rayons d'épicerie.
- Caméra : **FIXE** (diff 0,28, décalage 0,2 px).

| Fichier | Original | Contenu / usage |
|---|---|---|
| `web_merl_01_passage.mp4` (60 s) | `10_1_crop.mp4`, de 2 s à 62 s | Une personne vue de dessus prend des produits en rayon, les examine, les repose. |
| `web_merl_02_passage.mp4` (60 s) | `11_1_crop.mp4`, de 2 s à 62 s | Idem, autre sujet. |

Limite : les produits sont de petits emballages serrés les uns contre les autres, hors classes COCO ;
nous n'annotons pas de retrait (les produits sont reposés) : ces vidéos servent à vérifier que les
personnes sont bien suivies et qu'il n'y a pas d'alerte, et à essayer `roi_crops` + `world`.

## 4. Pexels (grand magasin, rayon rempli d'articles)

Téléchargée le 26 novembre 2026 parmi 14 clips candidats (Mixkit, Pexels) testés avec
`tools/check_camera.py` ; c'est le seul clip de grand magasin **à caméra fixe** avec un rayon
entièrement rempli d'articles et un client qui s'y arrête. Licence : « Pexels License »
(https://www.pexels.com/license/ : usage gratuit, attribution non obligatoire).

| Fichier | Titre | URL | Auteur | Licence | Caméra | Contenu / usage |
|---|---|---|---|---|---|---|
| `web_pexels_01_passage.mp4` (14,3 s) | Dairy Aisle Shopping in Supermarket | https://www.pexels.com/video/dairy-aisle-shopping-in-supermarket-39221961/ | (auteur affiché sur la page Pexels) | Pexels License | **FIXE** (diff 0,33, décalage 0,5 px) | Rayon produits laitiers d'un supermarché, des centaines de produits sur 5 étagères. Une cliente arrive à 7,7 s, longe le rayon en regardant les produits (main tendue vers 8 à 9,5 s), et sort du cadre à 13,4 s. Aucun retrait visible : test « grand magasin, beaucoup d'articles, passage » (aucune alerte attendue). Vitrines : `Rayon haut`, `Rayon bas`. |

Version téléchargée : 1920×1080, 25 img/s (https://www.pexels.com/download/video/39221961/),
convertie en 960 px à 30 img/s. Les autres clips candidats (13 le 26/11 pour le grand magasin, 25 pour le magasin de vêtements, 9 pour
une scène de vol ou d'objet pris sur une table : caméra mobile, articles illisibles ou scène sans intérêt) ont été supprimés après analyse.

### 4.2 Trois vidéos Pexels choisies par l'étudiante (26 novembre 2026)

Téléchargées à la main depuis Pexels (Pexels License, usage gratuit sans attribution obligatoire) ;
originaux 4K non conservés. Les trois sont filmées **caméra à la main** (verdict MOBILE de
`tools/check_camera.py`) : elles servent à tester la chaîne hors de ses hypothèses (caméra fixe).

| Fichier | Page Pexels | Original | Caméra | Contenu / usage |
|---|---|---|---|---|
| `web_pexels_02_retrait.mp4` (18,3 s) | https://www.pexels.com/video/8551735/ (« A Man Getting Cooking Oil in a Hypermarket Shop ») | 3840×2160, 25 img/s | **MOBILE** (diff 3,68, décalage 93 px : la caméra suit l'homme) | Rayon d'hypermarché ; un homme se baisse et prend une bouteille d'huile dans le rayon de droite vers 8,5 s, puis marche vers la caméra. Vitrines `Rayon huile`, `Rayon gauche`. `RETIRE` bouteille attendu. |
| `web_pexels_03_retrait.mp4` (29,8 s, portrait 540×1024) | https://www.pexels.com/video/6649434/ | 2160×4096, 25 img/s | **MOBILE** (diff 2,76, décalage 291 px : la caméra bascule vers le bas puis remonte) | Boutique de maroquinerie : sacs et portefeuilles sur étagères ; une main prend un portefeuille sur l'étagère du haut vers 6,5 s, l'examine, le repose vers 24 s. Vitrines `Etagere portefeuilles`, `Etagere basse`, `Sacs`. `RETIRE` puis `REMIS` attendus. |
| `web_pexels_04_retrait.mp4` (14,5 s) | https://www.pexels.com/video/12795687/ | 4096×2160, 25 img/s | **MOBILE** (diff 9,48, décalage 458 px : gros plan à la main) | Une main ouvre un congélateur de supermarché et prend un sachet surgelé vers 5,5 s. Vitrine `Congelateur` (toute l'image). Cas limite : caméra très mobile, articles identiques derrière une vitre. |

## 5. Vidéo du dépôt de référence

- `ref_kaustuv_bouteille.mp4` ← `reference/Missing-Object-Detection/Whatsapp Vid 2.mp4`
  (https://github.com/kaustuv123/Missing-Object-Detection, dépôt sans fichier de licence ; vidéo
  fournie par l'auteur pour tester son projet).
- 478×850 (portrait), 29,87 img/s, 14,5 s, convertie en 480×854 à 30 img/s.
- Caméra : **MOBILE** à la fin (téléphone tenu à la main, décalage jusqu'à 366 px dans les 3
  dernières secondes), à peu près stable avant.
- Contenu : une personne face caméra tient une bouteille qui apparaît (2,0 s), disparaît (5,3 s),
  réapparaît (7,5 s) et disparaît à nouveau (9,3 s). Sert à comparer nos événements à ceux du dépôt
  de référence.

## 6. Vidéo synthétique

- `synth_boutique.mp4` + `synth_boutique.zones.json`, générés par `python tools/make_shop_video.py`
  (OpenCV, 40 s, 30 img/s, 960×540). Vérité terrain exacte : `data/annotations_synth.csv`.
- Détection par couleur uniquement (`detector.type: simulated`).
