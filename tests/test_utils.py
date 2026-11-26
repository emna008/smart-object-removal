import csv

import numpy as np

from alerts.notifier import AlertNotifier
from engine.scene_monitor import Event
from engine.visual_check import crop_patch, patch_similarity
from engine.zone import Zone
from utils.geometry import coverage, iou


def test_iou_et_couverture():
    a = (0, 0, 10, 10)
    assert iou(a, a) == 1.0
    assert iou(a, (20, 20, 30, 30)) == 0.0
    assert coverage((0, 0, 10, 10), (0, 0, 5, 10)) == 0.5


def test_zone_polygone():
    zone = Zone([[0, 0], [100, 0], [100, 100], [0, 100]])
    assert zone.contains_box((10, 10, 30, 30))
    assert not zone.contains_box((150, 150, 170, 170))
    assert Zone(None).contains_box((5000, 5000, 5010, 5010))


def test_zone_sauvegarde_et_mise_a_l_echelle(tmp_path):
    path = tmp_path / "zone.json"
    Zone([[0, 0], [100, 0], [100, 50]]).save(str(path), frame_width=1000)
    zone = Zone.load(str(path), frame_width=500)
    assert zone.points.tolist() == [[0, 0], [50, 0], [50, 25]]


def test_similarite_visuelle():
    frame = np.full((200, 200, 3), 120, np.uint8)
    yy, xx = np.mgrid[0:100, 0:100]
    frame[50:150, 50:150] = (((yy // 12 + xx // 12) % 2) * 150 + 40)[..., None]  # motif de l'objet
    box = (50, 50, 150, 150)
    ref = crop_patch(frame, box)
    assert patch_similarity(ref, crop_patch(frame, box)) > 0.99
    brighter = np.clip(frame.astype(int) + 40, 0, 255).astype(np.uint8)
    assert patch_similarity(ref, crop_patch(brighter, box)) > 0.9      # même objet, plus de lumière
    empty = np.full_like(frame, 120)
    assert patch_similarity(ref, crop_patch(empty, box)) == 0.0         # emplacement vide


def test_journal_csv_sans_doublons(tmp_path):
    ev = Event(time_s=7.0, frame=70, type="RETIRE", ref_id=1, class_name="cup", track_id=3,
               bbox=(1, 2, 3, 4), lost_at_s=4.0, person_near=True)
    for _ in range(2):  # même vidéo traitée deux fois
        notifier = AlertNotifier(str(tmp_path), "clip01.mp4")
        notifier.register([ev], 7.0)
    with open(tmp_path / "events.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["evenement"] == "RETIRE" and rows[0]["personne_proche"] == "oui"
    assert rows[0]["capture"].endswith("_RETIRE_obj1.jpg")
