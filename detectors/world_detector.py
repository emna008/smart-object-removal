"""Détecteur à vocabulaire ouvert (`detector.type: world`) : YOLO-World ou YOLOE.

Les classes sont écrites en texte dans config.yaml (`detector.world_classes`), par
exemple ["shoe", "guitar", "handbag", ...] : utile pour les articles de boutique
qui ne sont pas dans COCO. Les personnes sont détectées par le même modèle en
ajoutant "person" au vocabulaire.

API vérifiée dans ultralytics 8.4.218 (voir README) :
  - YOLO-World : YOLOWorld("yolov8s-worldv2.pt").set_classes(["shoe", ...])
  - YOLOE      : m = YOLOE("yoloe-26n-seg.pt"); m.set_classes(names, m.get_text_pe(names))
Le codage du texte utilise CLIP (YOLO-World) ou MobileCLIP (YOLOE), téléchargés
au premier lancement. Tout tourne sur CPU.
"""
from typing import List, Tuple

import numpy as np

from detectors.base import BaseDetector
from detectors.yolo_detector import split_result
from utils.types import Detection

DEFAULT_WORLD_MODEL = "yolov8s-worldv2.pt"
DEFAULT_CLASSES = ["shoe", "sneaker", "guitar", "handbag", "watch", "perfume bottle", "sunglasses",
                   "t-shirt", "bottle", "box"]
PERSON = "person"


class WorldDetector(BaseDetector):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_path = config.get("world_model") or DEFAULT_WORLD_MODEL
        classes = list(config.get("world_classes") or DEFAULT_CLASSES)
        occluders = list(config.get("occluder_classes") or [PERSON])
        vocab = classes + [c for c in occluders if c not in classes]
        device = config.get("device", "auto")
        self.device = None if device in (None, "auto") else device

        print(f"[detecteur] chargement de {self.model_path} (vocabulaire ouvert) ...")
        if "yoloe" in self.model_path.lower():
            from ultralytics import YOLOE
            self.model = YOLOE(self.model_path)
            self.model.set_classes(vocab, self.model.get_text_pe(vocab))
        else:
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(self.model_path)
            self.model.set_classes(vocab)
        names = self.model.names
        self.names = {int(k): v for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
        by_name = {v: k for k, v in self.names.items()}
        self.target_ids = [by_name[c] for c in classes if c in by_name]
        self.occluder_ids = [by_name[c] for c in occluders if c in by_name]
        print("[detecteur] vocabulaire :", ", ".join(vocab))

    def _detect_raw(self, image: np.ndarray, want_targets: bool = True,
                    want_occluders: bool = True) -> Tuple[List[Detection], List[Detection]]:
        ids = sorted((set(self.target_ids) if want_targets else set())
                     | (set(self.occluder_ids) if want_occluders else set()))
        kwargs = dict(conf=self.confidence, imgsz=self.imgsz, classes=ids, verbose=False)
        if self.device is not None:
            kwargs["device"] = self.device
        result = self.model.predict(image, **kwargs)[0]
        return split_result(result, self.names, self.target_ids, self.occluder_ids)
