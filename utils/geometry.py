"""Petites fonctions géométriques sur les boîtes (x1, y1, x2, y2)."""
import math
from typing import Tuple

from utils.types import BBox


def box_center(b: BBox) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def box_area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_diag(b: BBox) -> float:
    """Diagonale de la boîte : sert d'échelle pour les seuils de distance."""
    return max(1.0, math.hypot(b[2] - b[0], b[3] - b[1]))


def intersection_area(a: BBox, b: BBox) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def iou(a: BBox, b: BBox) -> float:
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def coverage(a: BBox, by: BBox) -> float:
    """Part de la boîte `a` recouverte par la boîte `by` (entre 0 et 1)."""
    area = box_area(a)
    return intersection_area(a, by) / area if area > 0 else 0.0


def distance(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def expand_box(b: BBox, ratio: float, width: int, height: int) -> Tuple[int, int, int, int]:
    """Agrandit la boîte de `ratio` de chaque côté et la limite à l'image."""
    w, h = b[2] - b[0], b[3] - b[1]
    x1 = int(max(0, math.floor(b[0] - ratio * w)))
    y1 = int(max(0, math.floor(b[1] - ratio * h)))
    x2 = int(min(width, math.ceil(b[2] + ratio * w)))
    y2 = int(min(height, math.ceil(b[3] + ratio * h)))
    return x1, y1, x2, y2
