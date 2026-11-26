"""Alertes : console, bandeau à l'écran, journal CSV, captures d'écran et clips.

Chaque alerte RETIRE indique la vitrine, l'article et le client associé.
"""
import csv
import os
from typing import List

import cv2
import numpy as np

from engine.scene_monitor import EV_DEPLACE, EV_NOUVEL_OBJET, EV_REMIS, EV_RETIRE, Event

CSV_HEADER = ["video", "temps_s", "image", "evenement", "vitrine", "objet", "classe", "track_id",
              "personne_id", "x1", "y1", "x2", "y2", "disparu_a_s", "personne_proche", "capture", "clip",
              "details"]

BANNER_COLORS = {  # BGR
    EV_RETIRE: (40, 40, 220),
    EV_REMIS: (60, 160, 60),
    EV_DEPLACE: (0, 150, 230),
    EV_NOUVEL_OBJET: (150, 110, 40),
}
CAPTURE_EVENTS = (EV_RETIRE, EV_REMIS)
CLIP_EVENTS = (EV_RETIRE,)


class AlertNotifier:
    def __init__(self, out_dir: str, video_name: str, save_screenshots: bool = True,
                 banner_seconds: float = 3.0, csv_name: str = "events.csv", save_clips: bool = False):
        self.out_dir = out_dir
        self.video_name = video_name
        self.save_screenshots = save_screenshots
        self.save_clips = save_clips
        self.banner_seconds = banner_seconds
        self.capture_dir = os.path.join(out_dir, "captures")
        os.makedirs(self.capture_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, csv_name)
        self._purge_previous_rows()
        self.banners: List[tuple] = []   # (texte, couleur, fin)
        self.recent: List[Event] = []

    def _purge_previous_rows(self) -> None:
        """Relancer une vidéo remplace ses anciennes lignes (pas de doublons).
        Un ancien journal sans les nouvelles colonnes est complété."""
        rows = []
        if os.path.exists(self.csv_path):
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r.get("video") != self.video_name]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in CSV_HEADER})

    @staticmethod
    def describe(ev: Event) -> str:
        """Texte sans accent pour l'écran : vitrine, article, client."""
        text = f"{ev.type}: {ev.class_name} #{ev.ref_id}"
        if ev.zone_name:
            text += f" | {ev.zone_name}"
        if ev.person_id is not None:
            text += f" | client {ev.person_id}"
        return text

    def register(self, events: List[Event], t: float) -> None:
        """Affiche, journalise et prépare les bandeaux (avant le dessin de l'image)."""
        if not events:
            return
        stem = os.path.splitext(os.path.basename(self.video_name))[0]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for ev in events:
                if self.save_screenshots and ev.type in CAPTURE_EVENTS:
                    ev.capture = f"{stem}_{ev.frame:06d}_{ev.type}_obj{ev.ref_id}.jpg"
                if self.save_clips and ev.type in CLIP_EVENTS:
                    ev.clip = f"{stem}_{ev.frame:06d}_{ev.type}_obj{ev.ref_id}.mp4"
                x1, y1, x2, y2 = (round(v) for v in ev.bbox) if ev.bbox else ("", "", "", "")
                writer.writerow([
                    self.video_name, f"{ev.time_s:.2f}", ev.frame, ev.type, ev.zone_name, ev.ref_id,
                    ev.class_name, ev.track_id if ev.track_id is not None else "",
                    ev.person_id if ev.person_id is not None else "", x1, y1, x2, y2,
                    f"{ev.lost_at_s:.2f}" if ev.lost_at_s is not None else "",
                    {True: "oui", False: "non"}.get(ev.person_near, ""), ev.capture, ev.clip, ev.details,
                ])
                who = f", client {ev.person_id}" if ev.person_id is not None else ""
                print(f"[{ev.time_s:7.2f} s] {ev.type:<12} {ev.zone_name or '-'} : objet #{ev.ref_id} "
                      f"({ev.class_name}){who}{' - ' + ev.details if ev.details else ''}")
                self.recent.append(ev)
                if ev.type in BANNER_COLORS:
                    self.banners.append((self.describe(ev), BANNER_COLORS[ev.type], t + self.banner_seconds))
        self.recent = self.recent[-50:]

    def save_captures(self, events: List[Event], image: np.ndarray) -> None:
        for ev in events:
            if ev.capture:
                cv2.imwrite(os.path.join(self.capture_dir, ev.capture), image)

    def active_banners(self, t: float) -> List[tuple]:
        self.banners = [b for b in self.banners if b[2] > t]
        return self.banners[-3:]
