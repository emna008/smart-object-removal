"""Types de données partagés entre les modules."""
from dataclasses import dataclass
from typing import Tuple

# Boîte englobante (x1, y1, x2, y2) en pixels de l'image traitée
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """Une détection YOLO sur une image."""
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class Track:
    """Un objet suivi par le tracker, avec son identifiant stable."""
    track_id: int
    bbox: BBox
    class_name: str
    confidence: float = 0.0
