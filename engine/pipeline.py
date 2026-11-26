"""Boucle principale : vidéo -> détection -> suivi (articles, clients) -> moniteur
-> alertes -> affichage -> preuve vidéo."""
import json
import os
import time
from collections import Counter
from typing import List, Optional, Union

import cv2
import numpy as np

from engine.clip_buffer import ClipRecorder
from utils.geometry import iou
from utils.timer import FPSCounter
from utils.types import Detection, Track

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")


def resize_to_width(frame: np.ndarray, width: Optional[int]) -> np.ndarray:
    """Réduit l'image à `width` pixels de large ; une image plus étroite n'est jamais agrandie."""
    if not width or frame.shape[1] <= width:
        return frame
    scale = width / float(frame.shape[1])
    return cv2.resize(frame, (width, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def open_capture(source: Union[str, int]) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la source video : {source}")
    return cap


def read_first_frame(source: Union[str, int], width: Optional[int]) -> np.ndarray:
    cap = open_capture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"Aucune image lisible dans : {source}")
    return resize_to_width(frame, width)


def merge_persons(person_tracks: List[Track], detections: List[Detection]) -> list:
    """Pistes des clients + détections pas encore confirmées par le tracker
    (sans identifiant) : l'occultation fonctionne dès la première image."""
    merged: list = list(person_tracks)
    for d in detections:
        if all(iou(d.bbox, p.bbox) < 0.3 for p in person_tracks):
            merged.append(d)
    return merged


class Pipeline:
    def __init__(self, config: dict, detector, tracker, monitor, notifier, visualizer, person_tracker=None):
        self.config = config
        self.detector = detector
        self.tracker = tracker
        self.person_tracker = person_tracker
        self.monitor = monitor
        self.notifier = notifier
        self.visualizer = visualizer
        self.resize_width = (config.get("video") or {}).get("resize_width")
        out = config.get("output") or {}
        self.out_dir = out.get("dir", "outputs")
        self.save_clips = bool(out.get("save_clips", False))
        self.clip_pre_s = float(out.get("clip_pre_seconds", 5.0))
        self.clip_post_s = float(out.get("clip_post_seconds", 5.0))
        if hasattr(detector, "set_zones"):
            detector.set_zones(monitor.zones)

    def run(self, source: Union[str, int], video_name: str, show: bool = True,
            save_video: bool = False, max_frames: Optional[int] = None) -> dict:
        cap = open_capture(source)
        is_live = isinstance(source, int)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not is_live and video_fps <= 1:
            video_fps = 30.0
        writer = None
        clips = ClipRecorder(self.out_dir, video_fps or 30.0, self.clip_pre_s, self.clip_post_s) \
            if self.save_clips else None
        fps_counter = FPSCounter()
        fps = 0.0
        frame_idx = -1
        t = 0.0
        start_wall = time.perf_counter()
        paused = False
        window = "Smart Object Removal"
        view = None

        try:
            while True:
                if not paused:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_idx += 1
                    if max_frames is not None and frame_idx >= max_frames:
                        break
                    frame = resize_to_width(frame, self.resize_width)
                    t = (time.perf_counter() - start_wall) if is_live else frame_idx / video_fps

                    targets, person_dets = self.detector.detect(frame)
                    tracks = self.tracker.update(targets, frame)
                    if self.person_tracker is not None:
                        persons = merge_persons(self.person_tracker.update(person_dets, frame), person_dets)
                    else:
                        persons = person_dets
                    events = self.monitor.update(t, frame_idx, tracks, persons, frame)
                    self.notifier.register(events, t)
                    fps = fps_counter.tick()
                    view = self.visualizer.render(frame, self.monitor, tracks, persons,
                                                  self.notifier.active_banners(t), fps, t,
                                                  self.notifier.recent)
                    self.notifier.save_captures(events, view)
                    if clips is not None:
                        clips.push(t, view)
                        for ev in events:
                            if ev.clip:
                                clips.start(ev.clip, t)

                    if save_video:
                        if writer is None:
                            os.makedirs(os.path.join(self.out_dir, "videos"), exist_ok=True)
                            stem = os.path.splitext(os.path.basename(video_name))[0]
                            path = os.path.join(self.out_dir, "videos", f"{stem}_annotee.mp4")
                            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                                     video_fps or 20.0, (view.shape[1], view.shape[0]))
                        writer.write(view)

                if show and view is not None:
                    try:
                        cv2.imshow(window, view)
                        key = cv2.waitKey(1 if not paused else 50) & 0xFF
                    except cv2.error:
                        print("[affichage] fenetre indisponible (OpenCV sans interface) : affichage desactive")
                        show = False
                        continue
                    if key in (ord("q"), 27):
                        break
                    if key == ord("p"):
                        paused = not paused
                    elif key == ord("s"):
                        path = os.path.join(self.out_dir, "captures", f"manuel_{frame_idx:06d}.jpg")
                        cv2.imwrite(path, view)
                        print(f"[capture] {path}")
                    elif key == ord("r"):
                        self.monitor.reset()
                        print("[moniteur] inventaire reinitialise")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if clips is not None:
                clips.close()
            if show:
                cv2.destroyAllWindows()

        processing_s = time.perf_counter() - start_wall
        frames = frame_idx + 1
        counts = Counter(ev.type for ev in self.monitor.history)
        summary = {
            "video": video_name,
            "images": frames,
            "fps_video": round(video_fps, 2),
            "duree_video_s": round(t, 2),
            "duree_traitement_s": round(processing_s, 2),
            "fps_moyen_traitement": round(frames / processing_s, 2) if processing_s > 0 else 0.0,
            "detecteur": type(self.detector).__name__,
            "vitrines": self.monitor.zones.names,
            "evenements": dict(counts),
            "id_recuperes": self.monitor.stats["id_rebinds"],
            "occultations": self.monitor.stats["occlusions"],
            "sauvetages_visuels": self.monitor.stats["visual_saves"],
            "clients_suivis": len(self.monitor.persons),
            "clips": [os.path.basename(p) for p in clips.written] if clips is not None else [],
            "inventaire_final": [{"objet": r.ref_id, "classe": r.class_name, "vitrine": r.zone_name,
                                  "etat": r.state} for r in self.monitor.refs],
        }
        os.makedirs(os.path.join(self.out_dir, "resumes"), exist_ok=True)
        stem = os.path.splitext(os.path.basename(video_name))[0]
        with open(os.path.join(self.out_dir, "resumes", f"{stem}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary
