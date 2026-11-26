"""Preuve vidéo : tampon circulaire des dernières secondes et clips d'alerte.

À chaque RETIRE, on enregistre dans outputs/clips/ un clip annoté qui va de
`pre_seconds` avant l'alerte à `post_seconds` après. Les images du tampon sont
gardées compressées en JPEG pour limiter la mémoire (5 s à 30 img/s ~ 15 Mo).
"""
import os
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


class _Recording:
    def __init__(self, path: str, end_t: float, fps: float, size: Tuple[int, int]):
        self.path = path
        self.end_t = end_t
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, image: np.ndarray) -> None:
        self.writer.write(image)

    def close(self) -> None:
        self.writer.release()


class ClipRecorder:
    def __init__(self, out_dir: str, fps: float, pre_seconds: float = 5.0, post_seconds: float = 5.0,
                 jpeg_quality: int = 80):
        self.clip_dir = os.path.join(out_dir, "clips")
        self.fps = fps if fps and fps > 1 else 30.0
        self.pre_s = float(pre_seconds)
        self.post_s = float(post_seconds)
        self.quality = int(jpeg_quality)
        self.buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=int(round(self.pre_s * self.fps)) + 1)
        self.active: Dict[str, _Recording] = {}
        self.written: List[str] = []
        self._size: Optional[Tuple[int, int]] = None

    def push(self, t: float, image: np.ndarray) -> None:
        """À appeler pour chaque image annotée (après les captures)."""
        if self._size is None:
            self._size = (image.shape[1], image.shape[0])
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            self.buffer.append((t, buf))
        for name in list(self.active):
            rec = self.active[name]
            rec.write(image)
            if t >= rec.end_t:
                rec.close()
                self.written.append(rec.path)
                del self.active[name]

    def start(self, clip_name: str, t: float) -> Optional[str]:
        """Ouvre un clip : vide le tampon (avant l'alerte) puis continue `post_s` secondes."""
        if not clip_name or clip_name in self.active or self._size is None:
            return None
        os.makedirs(self.clip_dir, exist_ok=True)
        path = os.path.join(self.clip_dir, clip_name)
        rec = _Recording(path, t + self.post_s, self.fps, self._size)
        for _, buf in self.buffer:
            rec.write(cv2.imdecode(buf, cv2.IMREAD_COLOR))
        self.active[clip_name] = rec
        return path

    def close(self) -> List[str]:
        for rec in self.active.values():
            rec.close()
            self.written.append(rec.path)
        self.active.clear()
        return self.written
