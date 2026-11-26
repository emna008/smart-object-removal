"""Smart Object Removal — point d'entrée.

Exemples :
    python main.py --video videos/clip01.mp4
    python main.py --video videos/clip01.mp4 --select-zone      # dessiner les vitrines
    python main.py --video videos/ --no-show --save-video       # toutes les vidéos d'un dossier
    python main.py --video videos/synth_boutique.mp4 --detector simulated
    python main.py --video 0                                    # webcam
Touches : q quitter, p pause, s capture, r réinitialiser l'inventaire.
"""
import argparse
import os
import sys
from datetime import datetime

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from alerts.notifier import AlertNotifier  # noqa: E402
from detectors import DETECTOR_TYPES  # noqa: E402
from engine.pipeline import VIDEO_EXTENSIONS, Pipeline, read_first_frame  # noqa: E402
from engine.scene_monitor import SceneMonitor  # noqa: E402
from engine.zone import ZoneSet, resolve_path  # noqa: E402
from utils.visualizer import Visualizer  # noqa: E402

LEGACY_ZONE_FILE = "zone.json"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def list_sources(value):
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        return [int(value)]
    if os.path.isdir(value):
        files = sorted(f for f in os.listdir(value) if f.lower().endswith(VIDEO_EXTENSIONS))
        if not files:
            raise SystemExit(f"Aucune video dans le dossier {value}")
        return [os.path.join(value, f) for f in files]
    if not os.path.exists(value):
        raise SystemExit(f"Video introuvable : {value}")
    return [value]


def zones_file_for(source, cfg: dict, config_dir: str, must_exist: bool = True):
    """Ordre : <video>.zones.json à côté de la vidéo > zone.file de la config > ancien zone.json."""
    per_video = ZoneSet.file_for_video(source) if isinstance(source, str) else None
    if per_video:
        return per_video
    zone_cfg = cfg.get("zone") or {}
    candidates = [zone_cfg.get("file"), LEGACY_ZONE_FILE]
    for name in candidates:
        if not name:
            continue
        path = resolve_path(name, config_dir)
        if os.path.exists(path) or not must_exist:
            return path
    return None


def load_zones(source, cfg: dict, config_dir: str, frame_width: int) -> ZoneSet:
    path = zones_file_for(source, cfg, config_dir)
    if path:
        print(f"[vitrines] {path}")
        return ZoneSet.load(path, frame_width)
    points = (cfg.get("zone") or {}).get("points")
    if points:
        from engine.zone import Zone
        return ZoneSet([Zone(points)])
    return ZoneSet()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detection du retrait d'articles dans des vitrines surveillees")
    parser.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    parser.add_argument("--video", help="fichier video, dossier de videos, ou 0 pour la webcam")
    parser.add_argument("--select-zone", action="store_true",
                        help="dessiner les vitrines sur la premiere image (plusieurs polygones nommes)")
    parser.add_argument("--no-show", action="store_true", help="ne pas ouvrir de fenetre (traitement par lot)")
    parser.add_argument("--save-video", action="store_true", help="enregistrer la video annotee")
    parser.add_argument("--save-clips", action="store_true", help="enregistrer un clip de preuve a chaque RETIRE")
    parser.add_argument("--max-frames", type=int, help="limiter le nombre d'images (tests)")
    parser.add_argument("--model", help="remplace detector.model_path (ex. yolo26n.pt)")
    parser.add_argument("--detector", choices=DETECTOR_TYPES, help="remplace detector.type (coco, world, simulated)")
    parser.add_argument("--imgsz", type=int, help="remplace detector.imgsz (640 ou 1280)")
    parser.add_argument("--roi-crops", action="store_true", help="detecter les articles dans un recadrage de chaque vitrine")
    parser.add_argument("--delay", type=float, help="remplace monitor.removal_delay_seconds")
    parser.add_argument("--out", help="remplace output.dir (dossier des resultats)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    cfg.setdefault("detector", {})
    if args.model:
        cfg["detector"]["model_path"] = args.model
    if args.detector:
        cfg["detector"]["type"] = args.detector
    if args.imgsz:
        cfg["detector"]["imgsz"] = args.imgsz
    if args.roi_crops:
        cfg["detector"]["roi_crops"] = True
    if args.delay is not None:
        cfg.setdefault("monitor", {})["removal_delay_seconds"] = args.delay
    out_cfg = cfg.setdefault("output", {})
    if args.out:
        out_cfg["dir"] = args.out
    out_dir = resolve_path(out_cfg.get("dir", "outputs"), config_dir)
    out_cfg["dir"] = out_dir
    if args.save_clips:
        out_cfg["save_clips"] = True
    os.makedirs(out_dir, exist_ok=True)

    if args.video is not None:
        sources = list_sources(args.video)
    else:  # dans config.yaml, un chemin relatif part du dossier de la config
        configured = cfg["video"]["source"]
        if isinstance(configured, str) and not configured.isdigit():
            configured = resolve_path(configured, config_dir)
        sources = list_sources(configured)
    show = (cfg.get("display") or {}).get("show", True) and not args.no_show
    save_video = args.save_video or out_cfg.get("save_video", False)
    resize_width = (cfg.get("video") or {}).get("resize_width")

    first = read_first_frame(sources[0], resize_width)
    if args.select_zone:
        zones = ZoneSet.select_interactive(first)
        if isinstance(sources[0], str) and os.path.isfile(args.video or ""):
            zone_path = os.path.splitext(sources[0])[0] + ".zones.json"   # fichier propre a la video
        else:
            zone_path = resolve_path((cfg.get("zone") or {}).get("file") or "zones.json", config_dir)
        zones.save(zone_path, first.shape[1])
        print(f"[vitrines] {len(zones.zones)} enregistree(s) dans {zone_path} : {', '.join(zones.names)}")

    # Les imports lourds (PyTorch) seulement ici
    from detectors import build_detector
    from trackers.deepsort_tracker import DeepSORTTracker
    import torch
    torch.set_num_threads(int((cfg.get("video") or {}).get("threads") or os.cpu_count() or 4))

    detector = build_detector(cfg["detector"])
    person_cfg = cfg.get("person_tracker") or {}
    summaries = []
    for source in sources:
        if isinstance(source, int):
            name = f"webcam_{datetime.now():%Y%m%d_%H%M%S}"
            width = first.shape[1]
        else:
            name = os.path.basename(source)
            width = read_first_frame(source, resize_width).shape[1]
        print(f"\n=== {name} ===")
        zones = load_zones(source, cfg, config_dir, width)
        print("[vitrines]", "toute l'image" if zones.is_full_frame else ", ".join(zones.names))
        monitor = SceneMonitor(cfg.get("monitor"), zones)
        notifier = AlertNotifier(out_dir, name, out_cfg.get("save_screenshots", True),
                                 (cfg.get("display") or {}).get("banner_seconds", 3.0),
                                 save_clips=bool(out_cfg.get("save_clips", False)))
        person_tracker = DeepSORTTracker(person_cfg) if person_cfg.get("enabled", True) else None
        pipeline = Pipeline(cfg, detector, DeepSORTTracker(cfg["tracker"]), monitor, notifier,
                            Visualizer(cfg.get("display")), person_tracker=person_tracker)
        summary = pipeline.run(source, name, show=show, save_video=save_video, max_frames=args.max_frames)
        summaries.append(summary)
        print(f"-> {summary['images']} images, {summary['fps_moyen_traitement']} img/s, "
              f"evenements : {summary['evenements']}")

    print(f"\nJournal : {os.path.join(out_dir, 'events.csv')}")
    print("Evaluation : python tools/evaluate.py --annotations data/annotations.csv")


if __name__ == "__main__":
    main()
