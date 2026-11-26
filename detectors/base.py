"""Socle commun des détecteurs : recadrage par vitrine et dédoublonnage.

`detector.roi_crops: true` : les articles sont cherchés dans un recadrage agrandi
de chaque vitrine (utile pour une caméra en hauteur et de petits articles), puis
les boîtes sont remises dans les coordonnées de l'image complète. Les personnes
sont toujours détectées sur l'image entière.
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.geometry import iou
from utils.types import Detection

MAX_UPSCALE = 3.0


class BaseDetector:
    def __init__(self, config: dict):
        self.confidence = float(config.get("confidence", 0.35))
        self.imgsz = int(config.get("imgsz", 640))
        self.roi_crops = bool(config.get("roi_crops", False))
        self.roi_margin = float(config.get("roi_margin", 0.15))
        self.zones = None

    def set_zones(self, zones) -> None:
        """Les vitrines (ZoneSet) : nécessaires pour `roi_crops`."""
        self.zones = zones

    # À implémenter : (articles, personnes) sur une image donnée
    def _detect_raw(self, image: np.ndarray, want_targets: bool = True,
                    want_occluders: bool = True) -> Tuple[List[Detection], List[Detection]]:
        raise NotImplementedError

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], List[Detection]]:
        """Renvoie (articles surveillés, personnes)."""
        use_rois = self.roi_crops and self.zones is not None and not self.zones.is_full_frame
        targets, occluders = self._detect_raw(frame, want_targets=not use_rois, want_occluders=True)
        if not use_rois:
            return targets, occluders

        h, w = frame.shape[:2]
        for zone in self.zones.zones:
            x1, y1, x2, y2 = zone.bounding_rect(w, h, self.roi_margin)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crop = frame[y1:y2, x1:x2]
            scale = min(MAX_UPSCALE, self.imgsz / float(max(crop.shape[0], crop.shape[1])))
            if scale > 1.0:
                crop = cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)),
                                  interpolation=cv2.INTER_LINEAR)
            else:
                scale = 1.0
            found, _ = self._detect_raw(crop, want_targets=True, want_occluders=False)
            for d in found:
                bx1, by1, bx2, by2 = d.bbox
                full = (x1 + bx1 / scale, y1 + by1 / scale, x1 + bx2 / scale, y1 + by2 / scale)
                targets.append(Detection(full, d.confidence, d.class_id, d.class_name))
        return dedupe(targets), occluders


def dedupe(dets: List[Detection], iou_threshold: float = 0.6) -> List[Detection]:
    """Deux vitrines qui se recouvrent peuvent voir le même article : on garde le plus sûr."""
    kept: List[Detection] = []
    for d in sorted(dets, key=lambda x: -x.confidence):
        if all(not (k.class_name == d.class_name and iou(k.bbox, d.bbox) >= iou_threshold) for k in kept):
            kept.append(d)
    return kept


def filter_min_size(dets: List[Detection], min_side: float = 2.0) -> List[Detection]:
    return [d for d in dets if d.bbox[2] - d.bbox[0] >= min_side and d.bbox[3] - d.bbox[1] >= min_side]
