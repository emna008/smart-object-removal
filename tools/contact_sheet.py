"""Planche contact : une image toutes les `--step` secondes, tuilées avec l'instant.

Sert à annoter les vidéos (instant où un article quitte sa place) :
    python tools/contact_sheet.py videos/web_mixkit_01_retrait.mp4 --step 0.5 --out data/frames/
Produit <out>/<video>_planche.jpg (et, avec --frames, les images individuelles).
"""
import argparse
import math
import os

import cv2
import numpy as np


def extract(video_path: str, step_s: float, start_s: float = 0.0, end_s: float = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every = max(1, int(round(step_s * fps)))
    idx = 0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if end_s is not None and t > end_s:
            break
        if t >= start_s and idx % every == 0:
            frames.append((t, frame))
        idx += 1
    cap.release()
    return frames


def contact_sheet(frames, tile_width: int = 240, columns: int = 8) -> np.ndarray:
    tiles = []
    for t, frame in frames:
        scale = tile_width / frame.shape[1]
        tile = cv2.resize(frame, (tile_width, round(frame.shape[0] * scale)))
        label = f"{t:.1f}s"
        cv2.rectangle(tile, (0, 0), (62, 20), (0, 0, 0), -1)
        cv2.putText(tile, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        raise ValueError("aucune image")
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    rows = math.ceil(len(tiles) / columns)
    blank = np.zeros_like(tiles[0])
    grid = [np.hstack(tiles[r * columns:(r + 1) * columns] + [blank] * (columns - len(tiles[r * columns:(r + 1) * columns])))
            for r in range(rows)]
    return np.vstack(grid)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+")
    p.add_argument("--step", type=float, default=0.5)
    p.add_argument("--out", default="data/frames")
    p.add_argument("--tile-width", type=int, default=240)
    p.add_argument("--columns", type=int, default=8)
    p.add_argument("--frames", action="store_true", help="ecrire aussi chaque image")
    p.add_argument("--start", type=float, default=0.0, help="debut (s)")
    p.add_argument("--end", type=float, default=None, help="fin (s)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for video in args.videos:
        stem = os.path.splitext(os.path.basename(video))[0]
        frames = extract(video, args.step, args.start, args.end)
        sheet = contact_sheet(frames, args.tile_width, args.columns)
        path = os.path.join(args.out, f"{stem}_planche.jpg")
        cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{path} : {len(frames)} images, {sheet.shape[1]}x{sheet.shape[0]}")
        if args.frames:
            d = os.path.join(args.out, stem)
            os.makedirs(d, exist_ok=True)
            for t, f in frames:
                cv2.imwrite(os.path.join(d, f"{stem}_{t:05.1f}s.jpg"), f)


if __name__ == "__main__":
    main()
