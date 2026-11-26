"""Détecteur de test par couleur (`detector.type: simulated`).

YOLO ne reconnaît pas les dessins de la vidéo synthétique (tools/make_shop_video.py).
Ce détecteur retrouve chaque article par sa teinte (HSV) et les clients par la
couleur de leur silhouette. Il sert UNIQUEMENT à la vidéo synthétique et aux
tests : il n'a aucun intérêt sur une vraie vidéo.

La palette est partagée avec tools/make_shop_video.py pour que les deux restent
cohérents.
"""
from typing import Dict, List, Tuple

import cv2
import numpy as np

from detectors.base import BaseDetector, filter_min_size
from utils.types import Detection

# nom de classe -> teinte OpenCV (0-180). Les articles sont très saturés.
ITEM_HUES: Dict[str, int] = {
    "chaussure": 0,     # rouge
    "parfum": 30,       # jaune
    "sac": 60,          # vert
    "bijou": 90,        # cyan
    "montre": 120,      # bleu
    "lunettes": 150,    # magenta
}
PERSON_HUE = 15         # orange : silhouette des clients
PERSON_CLASS = "person"
HUE_TOL = 5
ITEM_SV = (230, 230)    # saturation, valeur de dessin des articles
PERSON_SV = (210, 170)


def bgr_from_hue(h: int, s: int, v: int) -> Tuple[int, int, int]:
    px = np.uint8([[[h, s, v]]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def item_color(name: str) -> Tuple[int, int, int]:
    return bgr_from_hue(ITEM_HUES[name], *ITEM_SV)


def person_color() -> Tuple[int, int, int]:
    return bgr_from_hue(PERSON_HUE, *PERSON_SV)


def hue_mask(hsv: np.ndarray, hue: int, tol: int, s_min: int, v_min: int) -> np.ndarray:
    lo, hi = hue - tol, hue + tol
    if lo < 0:
        m = cv2.inRange(hsv, (0, s_min, v_min), (hi, 255, 255)) | \
            cv2.inRange(hsv, (180 + lo, s_min, v_min), (180, 255, 255))
    elif hi > 180:
        m = cv2.inRange(hsv, (lo, s_min, v_min), (180, 255, 255)) | \
            cv2.inRange(hsv, (0, s_min, v_min), (hi - 180, 255, 255))
    else:
        m = cv2.inRange(hsv, (lo, s_min, v_min), (hi, 255, 255))
    return m


class ColorDetector(BaseDetector):
    def __init__(self, config: dict):
        super().__init__(config)
        wanted = config.get("color_classes") or list(ITEM_HUES)   # target_classes (COCO) est ignore ici
        unknown = [c for c in wanted if c not in ITEM_HUES]
        if unknown:
            raise ValueError(f"Classes inconnues pour le detecteur simule : {unknown}. "
                             f"Disponibles : {sorted(ITEM_HUES)}")
        self.classes = {name: ITEM_HUES[name] for name in wanted}
        self.min_area = int(config.get("min_area", 60))
        self.person_min_area = int(config.get("person_min_area", 400))
        self.s_min = int(config.get("s_min", 110))
        self.v_min = int(config.get("v_min", 80))
        self.kernel = np.ones((3, 3), np.uint8)
        print("[detecteur] simule (couleur) :", ", ".join(self.classes))

    def _blobs(self, mask: np.ndarray, min_area: int, class_id: int, name: str) -> List[Detection]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            conf = min(1.0, area / float(4 * min_area))
            out.append(Detection((float(x), float(y), float(x + w), float(y + h)), conf, class_id, name))
        return out

    def _detect_raw(self, image: np.ndarray, want_targets: bool = True,
                    want_occluders: bool = True) -> Tuple[List[Detection], List[Detection]]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        targets: List[Detection] = []
        occluders: List[Detection] = []
        if want_targets:
            for cid, (name, hue) in enumerate(self.classes.items()):
                targets += self._blobs(hue_mask(hsv, hue, HUE_TOL, self.s_min, self.v_min), self.min_area, cid, name)
        if want_occluders:
            occluders = self._blobs(hue_mask(hsv, PERSON_HUE, HUE_TOL, self.s_min, self.v_min - 20),
                                    self.person_min_area, 100, PERSON_CLASS)
        return filter_min_size(targets), occluders
