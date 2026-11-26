"""Zones surveillées : une ou plusieurs vitrines, chacune un polygone nommé.

Fichier de zones (nouveau format, écrit par --select-zone) :
    {"frame_width": 960,
     "zones": [{"nom": "Vitrine chaussures", "points": [[x, y], ...]}, ...]}
Formats aussi acceptés en lecture :
    - une liste nue [{"nom": ..., "points": ...}, ...] (sans mise à l'échelle) ;
    - l'ancien zone.json {"points": [...], "frame_width": W} : une seule zone,
      nommée « Zone 1 ».
Un fichier <video>.zones.json à côté de la vidéo est prioritaire sur zones.json.
"""
import json
import os
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

from utils.geometry import box_center
from utils.types import BBox

Points = List[List[int]]
DEFAULT_NAME = "Zone 1"
ZONE_COLORS = [(255, 200, 0), (0, 200, 255), (120, 220, 120), (230, 120, 230), (80, 160, 255), (255, 120, 120)]


class Zone:
    """Un polygone nommé. Sans points : toute l'image."""

    def __init__(self, points: Optional[Sequence[Sequence[float]]] = None, name: str = DEFAULT_NAME,
                 color=None):
        if points is not None and len(points) >= 3:
            self.points = np.array(points, dtype=np.int32)
        else:
            self.points = None  # pas de polygone : toute l'image est surveillée
        self.name = name
        self.color = color or ZONE_COLORS[0]

    @property
    def is_full_frame(self) -> bool:
        return self.points is None

    def contains_point(self, x: float, y: float) -> bool:
        if self.points is None:
            return True
        contour = self.points.reshape(-1, 1, 2)
        return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0

    def contains_box(self, bbox: BBox) -> bool:
        """Un objet est « dans la zone » si le centre de sa boîte y est."""
        return self.contains_point(*box_center(bbox))

    def bounding_rect(self, width: int, height: int, margin: float = 0.1):
        """Rectangle englobant (x1, y1, x2, y2) agrandi de `margin`, limité à l'image."""
        if self.points is None:
            return 0, 0, width, height
        x1, y1 = self.points.min(axis=0)
        x2, y2 = self.points.max(axis=0)
        mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
        return max(0, x1 - mx), max(0, y1 - my), min(width, x2 + mx), min(height, y2 + my)

    def draw(self, image: np.ndarray, color=None, alpha: float = 0.12, label: bool = True) -> None:
        if self.points is None:
            return
        color = color or self.color
        overlay = image.copy()
        cv2.fillPoly(overlay, [self.points], color)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)
        cv2.polylines(image, [self.points], True, color, 2, cv2.LINE_AA)
        if label:
            x, y = self.points.min(axis=0)
            cv2.putText(image, self.name, (int(x) + 6, int(y) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, self.name, (int(x) + 6, int(y) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 1, cv2.LINE_AA)

    def to_dict(self) -> dict:
        return {"nom": self.name, "points": self.points.tolist() if self.points is not None else None}

    # ------------------------------------------------------------- compatibilité
    def save(self, path: str, frame_width: int) -> None:
        ZoneSet([self]).save(path, frame_width)

    @classmethod
    def load(cls, path: str, frame_width: int) -> "Zone":
        """Ancienne API : renvoie la première zone du fichier."""
        zones = ZoneSet.load(path, frame_width)
        return zones.zones[0] if zones.zones else cls(None)

    @classmethod
    def select_interactive(cls, frame: np.ndarray, window: str = "Choisir la zone") -> "Zone":
        zones = ZoneSet.select_interactive(frame, window, single=True)
        return zones.zones[0] if zones.zones else cls(None)


class ZoneSet:
    """Les vitrines surveillées. Vide = toute l'image, sous le nom « Zone 1 »."""

    def __init__(self, zones: Optional[Iterable[Zone]] = None):
        self.zones: List[Zone] = [z for z in (zones or []) if not z.is_full_frame]
        for i, z in enumerate(self.zones):
            if z.color == ZONE_COLORS[0] and i > 0:
                z.color = ZONE_COLORS[i % len(ZONE_COLORS)]
        self._full = Zone(None, DEFAULT_NAME)

    @property
    def is_full_frame(self) -> bool:
        return not self.zones

    @property
    def names(self) -> List[str]:
        return [z.name for z in self.zones] if self.zones else [DEFAULT_NAME]

    def __len__(self) -> int:
        return len(self.zones) if self.zones else 1

    def __iter__(self):
        return iter(self.zones if self.zones else [self._full])

    def find(self, bbox: BBox) -> Optional[Zone]:
        """La vitrine qui contient le centre de la boîte, ou None."""
        if not self.zones:
            return self._full
        for z in self.zones:
            if z.contains_box(bbox):
                return z
        return None

    def name_of(self, bbox: BBox) -> Optional[str]:
        z = self.find(bbox)
        return z.name if z is not None else None

    def contains_box(self, bbox: BBox) -> bool:
        return self.find(bbox) is not None

    def get(self, name: str) -> Optional[Zone]:
        for z in self:
            if z.name == name:
                return z
        return None

    def draw(self, image: np.ndarray, alpha: float = 0.12) -> None:
        for z in self.zones:
            z.draw(image, alpha=alpha)

    # ------------------------------------------------------------------ fichiers
    def save(self, path: str, frame_width: int) -> None:
        data = {"frame_width": int(frame_width), "zones": [z.to_dict() for z in self.zones]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str, frame_width: int) -> "ZoneSet":
        """Charge les zones et les adapte si la largeur de traitement a changé."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):                       # liste nue, pas d'échelle connue
            raw, ref_width = data, frame_width
        elif "zones" in data:                            # nouveau format
            raw, ref_width = data["zones"], data.get("frame_width") or frame_width
        else:                                            # ancien zone.json : une seule zone
            raw = [{"nom": DEFAULT_NAME, "points": data.get("points")}]
            ref_width = data.get("frame_width") or frame_width
        scale = frame_width / float(ref_width) if ref_width and ref_width != frame_width else 1.0
        zones = []
        for i, item in enumerate(raw):
            pts = item.get("points")
            if not pts:
                continue
            if scale != 1.0:
                pts = [[round(x * scale), round(y * scale)] for x, y in pts]
            name = item.get("nom") or item.get("name") or f"Zone {i + 1}"
            zones.append(Zone(pts, name, ZONE_COLORS[i % len(ZONE_COLORS)]))
        return cls(zones)

    @staticmethod
    def file_for_video(video_path: str) -> Optional[str]:
        """<video>.zones.json à côté de la vidéo, s'il existe."""
        if not isinstance(video_path, str):
            return None
        stem, _ = os.path.splitext(video_path)
        candidate = stem + ".zones.json"
        return candidate if os.path.exists(candidate) else None

    # -------------------------------------------------------------- interactif
    @classmethod
    def select_interactive(cls, frame: np.ndarray, window: str = "Choisir les vitrines",
                           single: bool = False) -> "ZoneSet":
        """Dessine plusieurs polygones à la suite.
        Clic gauche : ajouter un point. Clic droit : retirer le dernier.
        Entrée : valider le polygone (le nom est demandé dans la console).
        Entrée sans point, ou Échap : terminer. R : recommencer le polygone courant."""
        points: Points = []
        zones: List[Zone] = []

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
            elif event == cv2.EVENT_RBUTTONDOWN and points:
                points.pop()

        cv2.namedWindow(window)
        cv2.setMouseCallback(window, on_mouse)
        help_lines = ["Clic gauche: ajouter un point | clic droit: annuler | R: recommencer",
                      "Entree: valider la vitrine (nom dans la console)",
                      "Entree sans point ou Echap: terminer (aucune vitrine = toute l'image)"]
        while True:
            view = frame.copy()
            for z in zones:
                z.draw(view)
            if len(points) >= 3:
                Zone(points, "", ZONE_COLORS[len(zones) % len(ZONE_COLORS)]).draw(view, label=False)
            for p in points:
                cv2.circle(view, tuple(p), 5, (0, 200, 255), -1, cv2.LINE_AA)
            if len(points) >= 2:
                cv2.polylines(view, [np.array(points, np.int32)], len(points) >= 3,
                              (0, 200, 255), 1, cv2.LINE_AA)
            for i, line in enumerate(help_lines):
                cv2.putText(view, line, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(view, line, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, view)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):
                if len(points) >= 3:
                    default = f"Vitrine {len(zones) + 1}"
                    try:
                        name = input(f"Nom de la vitrine [{default}] : ").strip() or default
                    except EOFError:
                        name = default
                    zones.append(Zone(list(points), name, ZONE_COLORS[len(zones) % len(ZONE_COLORS)]))
                    points.clear()
                    if single:
                        break
                else:
                    break
            if key == 27:
                break
            if key in (ord("r"), ord("R")):
                points.clear()
        cv2.destroyWindow(window)
        return cls(zones)


def resolve_path(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)
