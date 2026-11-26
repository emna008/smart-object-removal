"""Vitrines nommées et attribution à un client : tests sans modèle (détections simulées)."""
import json

import pytest

from engine.scene_monitor import EV_NOUVEL_OBJET, EV_REMIS, EV_RETIRE, RETIRE, SceneMonitor
from engine.zone import Zone, ZoneSet
from utils.types import Detection, Track

FPS = 10
BASE = {"visual_check": False}
V1 = Zone([[0, 0], [300, 0], [300, 300], [0, 300]], "Vitrine A")
V2 = Zone([[400, 0], [700, 0], [700, 300], [400, 300]], "Vitrine B")
SHOE_A = (100.0, 100.0, 140.0, 150.0)      # dans la vitrine A
BAG_B = (500.0, 100.0, 540.0, 150.0)       # dans la vitrine B


def run(monitor, until, scene):
    events = []
    for i in range(int(round(until * FPS)) + 1):
        t = round(i / FPS, 3)
        tracks, occ = scene(t)
        events += monitor.update(t, i, tracks, occ)
    return events


def of_type(events, ev_type):
    return [e for e in events if e.type == ev_type]


def shifted(box, dx, dy=0.0):
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def test_retrait_dans_une_seule_vitrine():
    m = SceneMonitor(BASE, ZoneSet([V1, V2]))

    def scene(t):
        tracks = [Track(2, BAG_B, "handbag", 0.9)]
        if t < 4.0:
            tracks.append(Track(1, SHOE_A, "shoe", 0.9))
        return tracks, []

    events = run(m, 10.0, scene)
    inv = of_type(events, "INVENTAIRE")
    assert sorted(e.zone_name for e in inv) == ["Vitrine A", "Vitrine B"]
    removed = of_type(events, EV_RETIRE)
    assert len(removed) == 1
    assert removed[0].zone_name == "Vitrine A" and removed[0].class_name == "shoe"
    assert [r.state for r in m.refs if r.zone_name == "Vitrine B"] == ["PRESENT"]


def test_association_du_bon_client_parmi_deux():
    m = SceneMonitor(BASE, ZoneSet([V1, V2]))
    far = Track(20, (600.0, 400.0, 660.0, 520.0), "person", 0.9)      # client loin, dans l'allée

    def scene(t):
        tracks = [Track(1, SHOE_A, "shoe", 0.9)] if t < 4.0 else []
        occ = [far]
        if 3.5 <= t < 5.0:                                             # client 21 recouvre la chaussure
            occ.append(Track(21, (90.0, 90.0, 160.0, 260.0), "person", 0.9))
        elif t >= 5.0:
            occ.append(Track(21, (90.0, 300.0, 160.0, 470.0), "person", 0.9))  # il s'éloigne
        return tracks, occ

    events = run(m, 12.0, scene)
    removed = of_type(events, EV_RETIRE)
    assert len(removed) == 1
    assert removed[0].person_id == 21
    assert 21 in m.persons and len(m.persons[21].trajectory) > 10


def test_article_passe_d_une_vitrine_a_l_autre():
    m = SceneMonitor({**BASE, "new_object_seconds": 1.0}, ZoneSet([V1, V2]))

    def scene(t):
        box = SHOE_A if t < 4.0 else shifted(SHOE_A, 400.0)   # même piste, désormais dans la vitrine B
        return [Track(1, box, "shoe", 0.9)], []

    events = run(m, 10.0, scene)
    removed = of_type(events, EV_RETIRE)
    new = of_type(events, EV_NOUVEL_OBJET)
    assert len(removed) == 1 and removed[0].zone_name == "Vitrine A"
    assert len(new) == 1 and new[0].zone_name == "Vitrine B"
    assert m.refs[0].state == RETIRE


def test_remis_dans_sa_vitrine_avec_client():
    m = SceneMonitor(BASE, ZoneSet([V1]))

    def scene(t):
        if t < 4.0:
            return [Track(1, SHOE_A, "shoe", 0.9)], []
        if t < 4.5:
            return [], [Track(7, (80.0, 80.0, 170.0, 280.0), "person", 0.9)]
        if t < 10.0:
            return [], []
        return [Track(9, shifted(SHOE_A, 2), "shoe", 0.9)], []

    events = run(m, 13.0, scene)
    assert len(of_type(events, EV_RETIRE)) == 1 and of_type(events, EV_RETIRE)[0].person_id == 7
    remis = of_type(events, EV_REMIS)
    assert len(remis) == 1 and remis[0].zone_name == "Vitrine A"


def test_detections_sans_identifiant_cachent_quand_meme():
    """Des Detection (sans track_id) jouent le rôle d'occultation ; pas de client associé."""
    m = SceneMonitor(BASE, ZoneSet([V1]))
    person = Detection((80.0, 80.0, 170.0, 280.0), 0.9, 0, "person")
    events = run(m, 12.0, lambda t: ([Track(1, SHOE_A, "shoe", 0.9)], []) if t < 4.0 else ([], [person]))
    assert of_type(events, EV_RETIRE) == []
    assert m.refs[0].state == "MASQUE"
    assert m.persons == {}


# ---------------------------------------------------------------- fichiers de zones
def test_ancien_zone_json_compatible(tmp_path):
    path = tmp_path / "zone.json"
    path.write_text(json.dumps({"points": [[0, 0], [100, 0], [100, 50]], "frame_width": 1000}), encoding="utf-8")
    zones = ZoneSet.load(str(path), frame_width=500)
    assert zones.names == ["Zone 1"]
    assert zones.zones[0].points.tolist() == [[0, 0], [50, 0], [50, 25]]
    legacy = Zone.load(str(path), frame_width=1000)      # ancienne API
    assert legacy.name == "Zone 1" and legacy.contains_box((10, 1, 30, 3))


def test_nouveau_format_et_liste_nue(tmp_path):
    path = tmp_path / "zones.json"
    ZoneSet([V1, V2]).save(str(path), frame_width=700)
    zones = ZoneSet.load(str(path), frame_width=350)
    assert zones.names == ["Vitrine A", "Vitrine B"]
    assert zones.name_of((10, 10, 20, 20)) == "Vitrine A"
    assert zones.name_of((210, 10, 220, 20)) == "Vitrine B"
    assert zones.find((160, 10, 170, 20)) is None          # entre les deux vitrines
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps([{"nom": "Etagere", "points": [[0, 0], [10, 0], [10, 10]]}]), encoding="utf-8")
    assert ZoneSet.load(str(bare), 960).names == ["Etagere"]


def test_fichier_de_zones_par_video(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    assert ZoneSet.file_for_video(str(video)) is None
    (tmp_path / "clip.zones.json").write_text("[]", encoding="utf-8")
    assert ZoneSet.file_for_video(str(video)).endswith("clip.zones.json")


def test_zone_vide_est_toute_l_image():
    zones = ZoneSet()
    assert zones.is_full_frame and zones.names == ["Zone 1"]
    assert zones.name_of((5000, 5000, 5010, 5010)) == "Zone 1"


def test_monitor_accepte_une_zone_simple():
    m = SceneMonitor(BASE, Zone([[0, 0], [250, 0], [250, 480], [0, 480]]))
    assert m.zones.names == ["Zone 1"]
    assert m.zone is m.zones


def test_deplacement_loin_avec_nouvel_identifiant_est_un_deplace():
    """L'article disparaît sous la main, puis réapparaît posé plus loin qu'une diagonale,
    sous un nouvel identifiant : DEPLACE, pas RETIRE ni NOUVEL_OBJET."""
    m = SceneMonitor(BASE, ZoneSet([V1]))
    hand = Track(5, (80.0, 80.0, 170.0, 280.0), "person", 0.9)

    def scene(t):
        if t < 4.0:
            return [Track(1, SHOE_A, "shoe", 0.9)], []
        if t < 5.0:
            return [], [hand]                                       # sous la main
        return [Track(8, shifted(SHOE_A, 120.0, 80.0), "shoe", 0.9)], []   # posé plus loin

    events = run(m, 12.0, scene)
    moved = of_type(events, "DEPLACE")
    assert len(moved) == 1 and 5.6 <= moved[0].time_s <= 6.0
    assert of_type(events, EV_RETIRE) == [] and of_type(events, EV_NOUVEL_OBJET) == []
    assert m.refs[0].track_id == 8 and m.refs[0].state == "PRESENT"
    assert m.refs[0].home_bbox == shifted(SHOE_A, 120.0, 80.0)


def test_article_porte_par_une_personne_n_est_pas_inventorie():
    """Lunettes sur un visage, t-shirt porté, bouteille tenue contre soi : pas des articles exposés."""
    m = SceneMonitor({**BASE, "new_object_seconds": 1.0}, ZoneSet([V1]))
    person = Track(3, (50.0, 20.0, 250.0, 300.0), "person", 0.9)
    worn = Track(1, (120.0, 60.0, 180.0, 90.0), "sunglasses", 0.9)         # entièrement dans la personne
    shelf = Track(2, (10.0, 200.0, 40.0, 240.0), "shoe", 0.9)              # hors de la personne
    events = run(m, 6.0, lambda t: ([worn, shelf], [person]))
    assert [e.class_name for e in events if e.type == "INVENTAIRE"] == ["shoe"]
    assert of_type(events, EV_NOUVEL_OBJET) == []


def test_retour_a_sa_place_apres_une_derive_n_est_pas_un_deplace():
    """La boîte suivie dérive loin de sa place (détection qui saute), disparaît, puis une
    nouvelle piste apparaît exactement à la place officielle : récupération, sans DEPLACE."""
    m = SceneMonitor(BASE, ZoneSet([V1]))

    def scene(t):
        if t < 4.0:
            return [Track(1, SHOE_A, "shoe", 0.9)], []
        if t < 4.3:
            return [Track(1, shifted(SHOE_A, 150.0, 100.0), "shoe", 0.9)], []   # dérive brève
        if t < 5.5:
            return [], []
        return [Track(4, shifted(SHOE_A, 2.0), "shoe", 0.9)], []              # de retour à sa place

    events = run(m, 9.0, scene)
    assert of_type(events, "DEPLACE") == [] and of_type(events, EV_RETIRE) == []
    assert of_type(events, EV_NOUVEL_OBJET) == []
    assert m.refs[0].track_id == 4 and m.refs[0].state == "PRESENT"


def test_limite_de_distance_pour_reconnaitre_un_article_deplace():
    """Même scénario que le DEPLACE sous nouvel identifiant, mais avec
    `rebind_max_distance_ratio: 2` : l'objet posé à 2,25 diagonales est trop loin pour
    être l'article manquant. Il devient NOUVEL_OBJET et la chaussure absente est RETIRE."""
    m = SceneMonitor(dict(BASE, rebind_max_distance_ratio=2.0), ZoneSet([V1]))
    hand = Track(5, (80.0, 80.0, 170.0, 280.0), "person", 0.9)

    def scene(t):
        if t < 4.0:
            return [Track(1, SHOE_A, "shoe", 0.9)], []
        if t < 5.0:
            return [], [hand]
        return [Track(8, shifted(SHOE_A, 120.0, 80.0), "shoe", 0.9)], []

    events = run(m, 12.0, scene)
    assert of_type(events, "DEPLACE") == []
    assert len(of_type(events, EV_NOUVEL_OBJET)) == 1
    removed = of_type(events, EV_RETIRE)
    assert len(removed) == 1 and removed[0].ref_id == 1


def test_meme_article_sous_un_autre_nom_n_est_pas_un_nouvel_objet():
    """Le détecteur nomme la même robe « dress » puis « shirt » (même boîte) : un seul article,
    pas de NOUVEL_OBJET ; une boîte voisine peu recouvrante reste un nouvel article."""
    m = SceneMonitor(BASE, ZoneSet([V1]))
    dress = (100.0, 60.0, 200.0, 260.0)
    nearby = (190.0, 60.0, 290.0, 260.0)     # recouvrement ~ 5 %

    def scene(t):
        tracks = [Track(1, dress, "dress", 0.9)]
        if t >= 4.0:
            tracks.append(Track(2, (102.0, 62.0, 198.0, 258.0), "shirt", 0.6))   # même robe, autre nom
        if t >= 8.0:
            tracks.append(Track(3, nearby, "handbag", 0.8))
        return tracks, []

    events = run(m, 12.0, scene)
    new = of_type(events, EV_NOUVEL_OBJET)
    assert [e.class_name for e in new] == ["handbag"]
    assert [r.class_name for r in m.refs] == ["dress", "handbag"]
