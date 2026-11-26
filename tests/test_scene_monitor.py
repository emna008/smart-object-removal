"""Tests de la logique du projet, sans caméra ni modèle : on simule ce que
renverraient YOLO + DeepSORT image par image (10 images par seconde)."""
import numpy as np
import pytest

from engine.scene_monitor import (ABSENT, EV_DEPLACE, EV_INVENTAIRE, EV_NOUVEL_OBJET, EV_REMIS, EV_RETIRE,
                                  MASQUE, PRESENT, RETIRE, SceneMonitor)
from engine.zone import Zone
from utils.types import Detection, Track

FPS = 10
CUP = (100.0, 100.0, 140.0, 150.0)
PERSON = Detection((60.0, 40.0, 320.0, 470.0), 0.9, 0, "person")
BASE = {"visual_check": False}


def cup(tid=1, box=CUP):
    return Track(tid, box, "cup", 0.9)


def shifted(box, dx, dy=0.0):
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def run(monitor, until, scene, frame_at=None):
    """scene(t) -> (tracks, occluders). Renvoie la liste des événements."""
    events = []
    for i in range(int(round(until * FPS)) + 1):
        t = round(i / FPS, 3)
        tracks, occluders = scene(t)
        frame = frame_at(t) if frame_at else None
        events += monitor.update(t, i, tracks, occluders, frame)
    return events


def of_type(events, ev_type):
    return [e for e in events if e.type == ev_type]


def test_inventaire_initial():
    m = SceneMonitor(BASE)
    events = run(m, 3.0, lambda t: ([cup()], []))
    inv = of_type(events, EV_INVENTAIRE)
    assert len(inv) == 1 and inv[0].class_name == "cup"
    assert inv[0].time_s == pytest.approx(2.0)
    assert m.refs[0].state == PRESENT


def test_retrait_alerte_apres_3_secondes():
    m = SceneMonitor(BASE)
    events = run(m, 10.0, lambda t: ([cup()] if t < 4.0 else [], []))
    removed = of_type(events, EV_RETIRE)
    assert len(removed) == 1
    assert 6.8 <= removed[0].time_s <= 7.1
    assert removed[0].lost_at_s == pytest.approx(4.0)
    assert m.refs[0].state == RETIRE


def test_main_qui_passe_pas_d_alerte():
    m = SceneMonitor(BASE)
    events = run(m, 10.0, lambda t: ([] if 4.0 <= t < 5.5 else [cup()], []))
    assert of_type(events, EV_RETIRE) == []
    assert m.refs[0].state == PRESENT


def test_objet_cache_par_une_personne_pas_d_alerte():
    m = SceneMonitor(BASE)
    states = []

    def scene(t):
        if 4.0 <= t < 12.0:
            return [], [PERSON]
        return [cup()], []

    events = []
    for i in range(int(14 * FPS) + 1):
        t = round(i / FPS, 3)
        tr, occ = scene(t)
        events += m.update(t, i, tr, occ)
        if m.refs:
            states.append(m.refs[0].state)
    assert of_type(events, EV_RETIRE) == []
    assert MASQUE in states
    assert m.stats["occlusions"] == 1


def test_personne_prend_l_objet_alerte_apres_son_depart():
    m = SceneMonitor(BASE)

    def scene(t):
        if t < 4.0:
            return [cup()], []
        if t < 9.0:
            return [], [PERSON]
        return [], []

    events = run(m, 14.0, scene)
    removed = of_type(events, EV_RETIRE)
    assert len(removed) == 1
    assert 11.8 <= removed[0].time_s <= 12.1
    assert removed[0].person_near is True


def test_occultation_trop_longue_finit_par_alerter():
    m = SceneMonitor({**BASE, "max_occlusion_seconds": 6.0})
    events = run(m, 16.0, lambda t: ([cup()], []) if t < 4.0 else ([], [PERSON]))
    removed = of_type(events, EV_RETIRE)
    assert len(removed) == 1
    assert 12.8 <= removed[0].time_s <= 13.1  # 4 + 6 s d'occultation + 3 s


def test_changement_d_identifiant_recupere():
    m = SceneMonitor(BASE)
    events = run(m, 10.0, lambda t: ([cup(1)] if t < 4.0 else [cup(7, shifted(CUP, 3))], []))
    assert of_type(events, EV_RETIRE) == []
    assert of_type(events, EV_NOUVEL_OBJET) == []
    assert m.stats["id_rebinds"] == 1
    assert m.refs[0].track_id == 7


def test_deplacement_dans_la_zone():
    m = SceneMonitor(BASE)

    def scene(t):
        dx = 0.0 if t < 4.0 else min(1.0, t - 4.0) * 300.0
        return [cup(1, shifted(CUP, dx))], []

    events = run(m, 10.0, scene)
    assert len(of_type(events, EV_DEPLACE)) == 1
    assert of_type(events, EV_RETIRE) == []
    assert m.refs[0].home_bbox == shifted(CUP, 300.0)


def test_sortie_de_la_zone_alerte():
    zone = Zone([[0, 0], [250, 0], [250, 480], [0, 480]])
    m = SceneMonitor(BASE, zone)

    def scene(t):
        dx = 0.0 if t < 4.0 else min(1.0, t - 4.0) * 300.0
        return [cup(1, shifted(CUP, dx))], []

    events = run(m, 12.0, scene)
    removed = of_type(events, EV_RETIRE)
    assert len(removed) == 1
    # le centre (120 px) passe 250 px vers t = 4.43 s, puis 3 s d'absence
    assert 7.3 <= removed[0].time_s <= 7.6


def test_nouvel_objet_pose_mais_pas_un_objet_transporte():
    m = SceneMonitor(BASE)
    book = (300.0, 200.0, 380.0, 260.0)

    def scene(t):
        tracks = [cup()]
        if t >= 5.0:
            tracks.append(Track(2, book, "book", 0.8))       # posé et immobile
        if 5.0 <= t < 12.0:
            x = 50.0 + (t - 5.0) * 60.0                       # bouteille transportée
            tracks.append(Track(3, (x, 300.0, x + 30.0, 380.0), "bottle", 0.8))
        return tracks, []

    events = run(m, 13.0, scene)
    new = of_type(events, EV_NOUVEL_OBJET)
    assert [e.class_name for e in new] == ["book"]
    assert 6.9 <= new[0].time_s <= 7.1


def test_objet_remis_a_sa_place():
    m = SceneMonitor(BASE)

    def scene(t):
        if t < 4.0:
            return [cup(1)], []
        if t < 10.0:
            return [], []
        return [cup(9, shifted(CUP, 2))], []

    events = run(m, 13.0, scene)
    assert len(of_type(events, EV_RETIRE)) == 1
    remis = of_type(events, EV_REMIS)
    assert len(remis) == 1 and 10.6 <= remis[0].time_s <= 10.9
    assert m.refs[0].state == PRESENT


def _frame_with_texture(textured: bool) -> np.ndarray:
    frame = np.full((480, 640, 3), 128, np.uint8)
    if textured:
        x1, y1, x2, y2 = map(int, CUP)
        yy, xx = np.mgrid[0:y2 - y1, 0:x2 - x1]
        frame[y1:y2, x1:x2] = (((yy // 8 + xx // 8) % 2) * 160 + 40)[..., None]  # motif de la tasse
    return frame


def test_verification_visuelle_evite_une_fausse_alerte():
    """Le détecteur rate l'objet à partir de 4 s, mais l'objet est toujours
    visible jusqu'à 10 s : pas d'alerte avant que l'emplacement soit vide."""
    m = SceneMonitor({"visual_check": True})
    textured, flat = _frame_with_texture(True), _frame_with_texture(False)
    events = run(m, 16.0, lambda t: ([cup()] if t < 4.0 else [], []),
                 frame_at=lambda t: textured if t < 10.0 else flat)
    removed = of_type(events, EV_RETIRE)
    assert m.stats["visual_saves"] >= 1
    assert len(removed) == 1 and removed[0].time_s > 10.0


def test_etat_absent_pendant_le_compte_a_rebours():
    m = SceneMonitor(BASE)
    run(m, 5.0, lambda t: ([cup()] if t < 4.0 else [], []))
    ref = m.refs[0]
    assert ref.state == ABSENT
    assert ref.absent_clock == pytest.approx(1.1, abs=0.05)
