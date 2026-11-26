"""Vérification visuelle : l'objet est-il encore visible à sa dernière position ?

Avant de déclencher une alerte, on compare l'image actuelle de l'emplacement de
l'objet avec une « photo » mémorisée quand l'objet était bien visible.
- Si les deux se ressemblent, l'objet est encore là : c'est le détecteur qui
  l'a raté (reflet, flou, lumière). Pas d'alerte.
- Si l'emplacement a changé (texture disparue), le retrait est confirmé.
"""
from typing import Optional

import cv2
import numpy as np

from utils.geometry import expand_box
from utils.types import BBox

PATCH_SIZE = 64          # les deux images sont ramenées à 64 × 64 pixels
TEXTURE_MIN_STD = 8.0    # en dessous, la zone est considérée comme unie


def crop_patch(frame: np.ndarray, bbox: BBox, pad: float = 0.1) -> Optional[np.ndarray]:
    """Découpe l'emplacement de la boîte, en niveaux de gris, légèrement flouté."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(bbox, pad, w, h)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.resize(gray, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def patch_similarity(reference: Optional[np.ndarray], current: Optional[np.ndarray]) -> Optional[float]:
    """Similarité entre 0 (différent) et 1 (identique), ou None si incomparable."""
    if reference is None or current is None:
        return None
    ref = reference.astype(np.float32)
    cur = current.astype(np.float32)
    ref_std, cur_std = float(ref.std()), float(cur.std())

    if ref_std >= TEXTURE_MIN_STD:
        if cur_std < TEXTURE_MIN_STD:
            return 0.0  # la texture de l'objet a disparu : emplacement vide
        # Corrélation normalisée : insensible aux changements de luminosité
        score = cv2.matchTemplate(cur, ref, cv2.TM_CCOEFF_NORMED)
        return float(max(0.0, score.max()))

    # Objet uni : on compare directement les intensités
    mad = float(np.abs(ref - cur).mean())
    return max(0.0, 1.0 - mad / 40.0)
