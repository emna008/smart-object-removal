"""Détecteur par couleur, recadrage par vitrine, clips de preuve et journal CSV."""
import os

import cv2
import numpy as np

from alerts.notifier import AlertNotifier
from detectors.base import BaseDetector, dedupe
from detectors.color_detector import ColorDetector, item_color, person_color
from engine.clip_buffer import ClipRecorder
from engine.scene_monitor import Event
from engine.zone import Zone, ZoneSet
from utils.types import Detection


def scene_image():
    img = np.full((300, 600, 3), (235, 235, 235), np.uint8)
    cv2.rectangle(img, (50, 50), (90, 90), item_color("chaussure"), -1)     # rouge
    cv2.circle(img, (300, 70), 20, item_color("sac"), -1)                    # vert
    cv2.ellipse(img, (450, 220), (40, 30), 0, 0, 360, person_color(), -1)    # client
    return img


def test_detecteur_couleur_trouve_articles_et_client():
    det = ColorDetector({"color_classes": ["chaussure", "sac"]})
    targets, persons = det.detect(scene_image())
    names = sorted(d.class_name for d in targets)
    assert names == ["chaussure", "sac"]
    shoe = next(d for d in targets if d.class_name == "chaussure")
    assert abs(shoe.bbox[0] - 50) <= 1 and abs(shoe.bbox[2] - 91) <= 1
    assert len(persons) == 1 and persons[0].class_name == "person"


class _FakeDetector(BaseDetector):
    """Enregistre les images reçues pour vérifier le recadrage."""

    def __init__(self, config):
        super().__init__(config)
        self.calls = []

    def _detect_raw(self, image, want_targets=True, want_occluders=True):
        self.calls.append((image.shape, want_targets, want_occluders))
        if want_targets and image.shape[1] < 600:          # recadrage : un article au centre du recadrage
            h, w = image.shape[:2]
            return [Detection((w / 2 - 5, h / 2 - 5, w / 2 + 5, h / 2 + 5), 0.9, 0, "shoe")], []
        return [], [Detection((0, 0, 10, 10), 0.9, 1, "person")] if want_occluders else []


def test_roi_crops_remet_les_boites_dans_l_image_complete():
    det = _FakeDetector({"roi_crops": True, "imgsz": 640, "roi_margin": 0.0})
    det.set_zones(ZoneSet([Zone([[100, 100], [200, 100], [200, 200], [100, 200]], "V")]))
    frame = np.zeros((400, 800, 3), np.uint8)
    targets, persons = det.detect(frame)
    assert len(persons) == 1                                   # personnes : image entière
    assert len(targets) == 1
    cx = (targets[0].bbox[0] + targets[0].bbox[2]) / 2
    cy = (targets[0].bbox[1] + targets[0].bbox[3]) / 2
    assert abs(cx - 150) < 2 and abs(cy - 150) < 2             # centre de la vitrine, coordonnées image
    assert det.calls[0][0] == (400, 800, 3) and det.calls[0][1] is False
    assert det.calls[1][0][0] == 300 and det.calls[1][0][1] == 300   # recadrage 100 px agrandi x3


def test_dedoublonnage():
    a = Detection((0, 0, 10, 10), 0.5, 0, "shoe")
    b = Detection((1, 1, 11, 11), 0.9, 0, "shoe")
    c = Detection((50, 50, 60, 60), 0.4, 0, "shoe")
    kept = dedupe([a, b, c])
    assert [d.confidence for d in kept] == [0.9, 0.4]


def test_clip_de_preuve(tmp_path):
    rec = ClipRecorder(str(tmp_path), fps=10, pre_seconds=1.0, post_seconds=0.5)
    img = np.zeros((40, 60, 3), np.uint8)
    for i in range(30):
        t = i / 10
        rec.push(t, img)
        if i == 20:
            rec.start("alerte.mp4", t)
    rec.close()
    path = os.path.join(str(tmp_path), "clips", "alerte.mp4")
    assert os.path.exists(path)
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert 15 <= n <= 18   # 11 images de tampon + 6 images après


def test_journal_csv_avec_vitrine_client_et_clip(tmp_path):
    ev = Event(time_s=7.0, frame=70, type="RETIRE", ref_id=1, class_name="shoe", track_id=3,
               bbox=(1, 2, 3, 4), lost_at_s=4.0, person_near=True, zone_name="Vitrine A", person_id=5)
    notifier = AlertNotifier(str(tmp_path), "clip01.mp4", save_clips=True)
    notifier.register([ev], 7.0)
    import csv
    with open(tmp_path / "events.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["vitrine"] == "Vitrine A" and rows[0]["personne_id"] == "5"
    assert rows[0]["clip"].endswith("_RETIRE_obj1.mp4")
    assert "client 5" in notifier.describe(ev) and "Vitrine A" in notifier.describe(ev)
