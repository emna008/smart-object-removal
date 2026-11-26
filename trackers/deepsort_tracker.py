"""Suivi multi-objets avec DeepSORT (bibliothèque deep-sort-realtime).

Choix repris du dépôt kaustuv123/Missing-Object-Detection (YOLOv8 + DeepSORT),
mais avec un vrai DeepSORT : en plus du filtre de Kalman et de l'IoU, chaque
objet a un descripteur d'apparence (ReID, réseau MobileNetV2). Quand une main
cache l'objet puis s'en va, l'apparence permet de lui rendre le même
identifiant au lieu d'en créer un nouveau.

`embedder: histogram` remplace MobileNetV2 par un histogramme de couleurs
(plus léger, utile pour comparer les deux approches dans le rapport).
`embedder_size` (défaut 128) : côté des vignettes données au ReID (la bibliothèque
utilise 224, trop lent sur CPU quand deux trackers tournent en même temps).
"""
from typing import List, Optional, Sequence

import cv2
import numpy as np

from utils.types import Detection, Track


class DeepSORTTracker:
    def __init__(self, config: dict):
        from deep_sort_realtime.deepsort_tracker import DeepSort

        embedder = config.get("embedder", "mobilenet")
        self.use_histogram = embedder in (None, "histogram", "none")
        gpu = bool(config.get("embedder_gpu", False))
        if not self.use_histogram:
            # Taille d'entrée du ReID MobileNetV2 (224 par défaut dans la bibliothèque) : 128 px
            # divise par ~3 le temps de calcul sur CPU pour des pertes de ré-identification minimes
            import deep_sort_realtime.embedder.embedder_pytorch as embedder_module
            embedder_module.INPUT_WIDTH = int(config.get("embedder_size", 128))
        self.tracker = DeepSort(
            max_age=int(config.get("max_age", 90)),
            n_init=int(config.get("n_init", 3)),
            max_iou_distance=float(config.get("max_iou_distance", 0.7)),
            max_cosine_distance=float(config.get("max_cosine_distance", 0.3)),
            nn_budget=config.get("nn_budget", 100),
            embedder=None if self.use_histogram else embedder,
            half=gpu,
            bgr=True,
            embedder_gpu=gpu,
        )

    def update(self, detections: Sequence[Detection], frame: np.ndarray) -> List[Track]:
        """Met à jour le suivi et renvoie les objets confirmés vus sur cette image."""
        dets = [d for d in detections if d.bbox[2] - d.bbox[0] > 1 and d.bbox[3] - d.bbox[1] > 1]
        raw = [([d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]], d.confidence, d.class_name)
               for d in dets]
        embeds: Optional[list] = None
        if self.use_histogram:
            embeds = [color_histogram(frame, d.bbox) for d in dets]
        tracks = self.tracker.update_tracks(raw, embeds=embeds, frame=frame)

        out: List[Track] = []
        for t in tracks:
            # on ne garde que les pistes confirmées ET associées à une détection maintenant
            if not t.is_confirmed() or t.time_since_update > 0:
                continue
            l, tp, r, b = t.to_ltrb(orig=True)
            out.append(Track(track_id=int(t.track_id), bbox=(float(l), float(tp), float(r), float(b)),
                             class_name=t.get_det_class(), confidence=float(t.get_det_conf() or 0.0)))
        return out


def color_histogram(frame: np.ndarray, bbox, bins: int = 8) -> np.ndarray:
    """Descripteur d'apparence simple : histogramme HSV normalisé."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
    x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
    crop = frame[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, bins], [0, 180, 0, 256, 0, 256]).flatten()
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist
