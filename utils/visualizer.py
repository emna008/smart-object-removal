"""Dessin de la vidéo annotée et du panneau d'inventaire par vitrine.

OpenCV n'affiche pas les accents : les textes à l'écran sont écrits sans accent.
"""
from typing import List, Optional, Sequence

import cv2
import numpy as np

from engine.scene_monitor import ABSENT, MASQUE, PRESENT, RETIRE, SceneMonitor
from utils.types import Detection, Track

FONT = cv2.FONT_HERSHEY_SIMPLEX
STATE_COLORS = {  # BGR
    PRESENT: (70, 190, 60),
    MASQUE: (0, 160, 255),
    ABSENT: (0, 215, 255),
    RETIRE: (50, 50, 230),
}
COLOR_CANDIDATE = (170, 170, 170)
COLOR_PERSON = (220, 150, 60)
COLOR_TAKER = (255, 80, 200)
COLOR_OTHER = (120, 120, 120)
PANEL_BG = (32, 28, 24)
TRAJECTORY_SECONDS = 6.0   # trajectoire du client dessinee apres l'alerte


def put_text(img, text, org, scale=0.55, color=(255, 255, 255), thickness=1, bg: Optional[tuple] = None):
    (w, h), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base + 1), bg, -1)
    else:
        cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def dashed_rect(img, p1, p2, color, thickness=2, dash=10):
    x1, y1 = map(int, p1)
    x2, y2 = map(int, p2)
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_trajectory(img, pts, color, thickness=2):
    pts = [(int(x), int(y)) for _, x, y in pts]
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, np.int32)], False, color, thickness, cv2.LINE_AA)
        cv2.arrowedLine(img, pts[max(0, len(pts) - 6)], pts[-1], color, thickness, cv2.LINE_AA, tipLength=0.4)


class Visualizer:
    def __init__(self, config: Optional[dict] = None):
        c = config or {}
        self.panel_width = int(c.get("panel_width", 380))

    def render(self, frame: np.ndarray, monitor: SceneMonitor, tracks: Sequence[Track],
               occluders: Sequence, banners: List[tuple], fps: float, t: float,
               recent_events: Sequence = ()) -> np.ndarray:
        img = frame.copy()
        monitor.zones.draw(img)

        # Clients (pistes du tracker des personnes, ou simples detections)
        takers = {r.taker_id for r in monitor.refs if r.taker_id is not None and r.state != PRESENT}
        for occ in occluders:
            x1, y1, x2, y2 = map(int, occ.bbox)
            pid = getattr(occ, "track_id", None)
            color = COLOR_TAKER if pid in takers else COLOR_PERSON
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1 if pid not in takers else 2)
            label = f"client {pid}" if pid is not None else getattr(occ, "class_name", "person")
            put_text(img, label, (x1 + 4, y1 + 18), 0.5, color)

        # Trajectoire du client associe a un retrait recent
        for ref in monitor.refs:
            if ref.state == RETIRE and ref.taker_id is not None and ref.removed_t is not None \
                    and t - ref.removed_t <= TRAJECTORY_SECONDS:
                person = monitor.persons.get(ref.taker_id)
                if person is not None:
                    draw_trajectory(img, person.trajectory, COLOR_TAKER, 2)

        known = {r.track_id for r in monitor.refs if r.state != RETIRE} | set(monitor.candidates)
        for tr in tracks:
            if tr.track_id not in known:
                x1, y1, x2, y2 = map(int, tr.bbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_OTHER, 1)

        for cand in monitor.candidates.values():
            if cand.last_t != t:
                continue
            x1, y1, x2, y2 = map(int, cand.bbox)
            dashed_rect(img, (x1, y1), (x2, y2), COLOR_CANDIDATE, 1, 6)
            put_text(img, f"{cand.class_name} ? {t - cand.first_t:.1f}s", (x1, y1 - 6), 0.5, COLOR_CANDIDATE)

        for ref in monitor.refs:
            self._draw_ref(img, ref, monitor)

        self._draw_hud(img, monitor, fps, t)
        self._draw_banners(img, banners)
        panel = self._draw_panel(img.shape[0], monitor, recent_events)
        return np.hstack([img, panel])

    def _draw_ref(self, img, ref, monitor):
        color = STATE_COLORS[ref.state]
        x1, y1, x2, y2 = map(int, ref.last_bbox)
        label = f"#{ref.ref_id} {ref.class_name} {ref.state}"
        if ref.state == ABSENT:
            label += f" {ref.absent_clock:.1f}/{monitor.removal_delay_s:.0f}s"
        if ref.state in (MASQUE, ABSENT, RETIRE) and ref.taker_id is not None:
            label += f" (client {ref.taker_id})"

        if ref.state == RETIRE:
            draw_trajectory(img, ref.trajectory, color, 2)
            dashed_rect(img, (x1, y1), (x2, y2), color, 2)
        elif ref.state == PRESENT:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        else:
            dashed_rect(img, (x1, y1), (x2, y2), color, 2, 8)
        put_text(img, label, (x1, max(18, y1 - 8)), 0.55, (255, 255, 255), 1, bg=color)

    def _draw_hud(self, img, monitor, fps, t):
        put_text(img, f"t = {t:.1f} s | {fps:.1f} img/s", (14, 28), 0.6, (255, 255, 255), bg=PANEL_BG)
        if not monitor.warmed_up:
            p = monitor.warmup_progress(t)
            put_text(img, f"Memorisation de l'inventaire... {int(p * 100)} %", (14, 58), 0.6,
                     (0, 215, 255), bg=PANEL_BG)

    def _draw_banners(self, img, banners):
        y = 70
        for text, color, _ in banners:
            scale = 0.8 if len(text) < 40 else 0.6
            (w, h), _ = cv2.getTextSize(text, FONT, scale, 2)
            x = max(10, (img.shape[1] - w) // 2)
            cv2.rectangle(img, (x - 12, y - h - 12), (x + w + 12, y + 12), color, -1)
            cv2.putText(img, text, (x, y), FONT, scale, (255, 255, 255), 2, cv2.LINE_AA)
            y += h + 34

    def _draw_panel(self, height, monitor, recent_events):
        panel = np.full((height, self.panel_width, 3), PANEL_BG, np.uint8)
        put_text(panel, "INVENTAIRE DES VITRINES", (16, 32), 0.6, (230, 230, 230), 1, bg=PANEL_BG)
        y = 64
        if not monitor.refs:
            put_text(panel, "(aucun article memorise)", (16, y), 0.5, (150, 150, 150), bg=PANEL_BG)
            y += 26
        shown = 0
        for zone in monitor.zones:
            refs = monitor.refs_in_zone(zone.name)
            if not refs and not monitor.zones.is_full_frame:
                continue
            if y > height - 140:
                break
            put_text(panel, zone.name, (16, y), 0.5, zone.color, bg=PANEL_BG)
            y += 24
            for ref in refs:
                if shown >= 14 or y > height - 140:
                    break
                color = STATE_COLORS[ref.state]
                cv2.circle(panel, (30, y - 5), 6, color, -1, cv2.LINE_AA)
                put_text(panel, f"#{ref.ref_id} {ref.class_name}", (44, y), 0.5, (230, 230, 230), bg=PANEL_BG)
                state = ref.state
                if ref.state == RETIRE and ref.taker_id is not None:
                    state += f" c{ref.taker_id}"
                put_text(panel, state, (self.panel_width - 118, y), 0.5, color, bg=PANEL_BG)
                y += 24
                shown += 1
            y += 6
        y += 10
        cv2.line(panel, (16, y), (self.panel_width - 16, y), (80, 80, 80), 1)
        y += 30
        put_text(panel, "DERNIERS EVENEMENTS", (16, y), 0.6, (230, 230, 230), bg=PANEL_BG)
        y += 30
        for ev in list(recent_events)[-8:][::-1]:
            if y > height - 36:
                break
            who = f" c{ev.person_id}" if ev.person_id is not None else ""
            put_text(panel, f"{ev.time_s:6.1f}s {ev.type} #{ev.ref_id} {ev.class_name}{who}", (16, y), 0.45,
                     (200, 200, 200), bg=PANEL_BG)
            y += 22
        stats = monitor.stats
        n_clients = len(monitor.persons)
        put_text(panel, f"ID recuperes: {stats['id_rebinds']}  occultations: {stats['occlusions']}  "
                        f"clients: {n_clients}", (16, height - 16), 0.42, (150, 150, 150), bg=PANEL_BG)
        return panel
