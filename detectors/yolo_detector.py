"""Détection d'objets avec YOLO (Ultralytics), modèle préentraîné sur COCO
(`detector.type: coco`).

Choix repris du dépôt kaustuv123/Missing-Object-Detection : YOLOv8n, le plus
léger de la famille YOLOv8, qui tourne en temps réel sur CPU. Le fichier de
poids est téléchargé automatiquement au premier lancement.
Autre option testable sans changer le code : `model_path: yolo26n.pt`.
"""
from typing import Iterable, List, Tuple

import numpy as np

from detectors.base import BaseDetector
from utils.types import Detection


class YOLODetector(BaseDetector):
    def __init__(self, config: dict):
        from ultralytics import YOLO  # import ici : les tests n'en ont pas besoin

        super().__init__(config)
        self.model_path = config.get("model_path", "yolov8n.pt")
        device = config.get("device", "auto")
        self.device = None if device in (None, "auto") else device

        print(f"[detecteur] chargement de {self.model_path} ...")
        self.model = YOLO(self.model_path)
        self.names = {int(k): v for k, v in self.model.names.items()}
        self.target_ids = self._class_ids(config.get("target_classes") or [])
        self.occluder_ids = self._class_ids(config.get("occluder_classes") or ["person"])
        if not self.target_ids:
            raise ValueError("detector.target_classes est vide : indiquez au moins une classe a surveiller")
        print("[detecteur] objets surveilles :", ", ".join(self.names[i] for i in self.target_ids))

    def _class_ids(self, names: Iterable[str]) -> List[int]:
        by_name = {v: k for k, v in self.names.items()}
        ids = []
        for name in names:
            if name not in by_name:
                raise ValueError(f"Classe inconnue '{name}'. Classes disponibles : {sorted(by_name)}")
            ids.append(by_name[name])
        return ids

    def _detect_raw(self, image: np.ndarray, want_targets: bool = True,
                    want_occluders: bool = True) -> Tuple[List[Detection], List[Detection]]:
        ids = sorted((set(self.target_ids) if want_targets else set())
                     | (set(self.occluder_ids) if want_occluders else set()))
        kwargs = dict(conf=self.confidence, imgsz=self.imgsz, classes=ids, verbose=False)
        if self.device is not None:
            kwargs["device"] = self.device
        result = self.model.predict(image, **kwargs)[0]
        return split_result(result, self.names, self.target_ids, self.occluder_ids)


def split_result(result, names: dict, target_ids, occluder_ids) -> Tuple[List[Detection], List[Detection]]:
    """Sépare un résultat Ultralytics en (articles, personnes)."""
    targets: List[Detection] = []
    occluders: List[Detection] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return targets, occluders
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
        det = Detection((float(x1), float(y1), float(x2), float(y2)), float(conf), int(cls), names[int(cls)])
        if cls in target_ids:
            targets.append(det)
        if cls in occluder_ids:
            occluders.append(det)
    return targets, occluders
