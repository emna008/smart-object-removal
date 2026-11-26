"""Caméra fixe ou non ? Mesure du mouvement de fond d'une vidéo.

Pour chaque paire d'images consécutives (réduites en gris), l'image est découpée
en blocs ; on prend la MÉDIANE des différences moyennes par bloc : les blocs où
une personne bouge sont rares et n'influencent pas la médiane, qui ne reflète
donc que le fond (bruit + mouvement de caméra). On estime aussi la translation
globale entre la première image et chaque image suivante (corrélation de phase).

    python tools/check_camera.py videos/*.mp4
Verdict : fixe si la médiane des différences de fond < 2.5 niveaux de gris et le
déplacement maximal < 1 % de la largeur.
"""
import argparse
import os

import cv2
import numpy as np

WIDTH = 320
BLOCK = 32


def analyse(path: str, sample_every: int = 2):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(path)
    prev = None
    first = None
    bg_diffs = []
    shifts = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (WIDTH, round(g.shape[0] * WIDTH / g.shape[1])))
            g = cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)
            if first is None:
                first = g
            if prev is not None:
                diff = np.abs(g - prev)
                h, w = diff.shape
                blocks = [diff[y:y + BLOCK, x:x + BLOCK].mean()
                          for y in range(0, h - BLOCK + 1, BLOCK) for x in range(0, w - BLOCK + 1, BLOCK)]
                bg_diffs.append(float(np.median(blocks)))
                (dx, dy), _ = cv2.phaseCorrelate(first, g)
                shifts.append(float(np.hypot(dx, dy)))
            prev = g
        idx += 1
    cap.release()
    return float(np.mean(bg_diffs)) if bg_diffs else 0.0, float(np.max(shifts)) if shifts else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+")
    p.add_argument("--max-diff", type=float, default=2.5)
    p.add_argument("--max-shift", type=float, default=0.01, help="fraction de la largeur")
    args = p.parse_args()
    print(f"{'video':45s} {'diff_fond':>10s} {'decalage_px':>12s}  verdict  (largeur analyse {WIDTH} px)")
    for v in args.videos:
        diff, shift = analyse(v)
        fixed = diff < args.max_diff and shift < args.max_shift * WIDTH
        print(f"{os.path.basename(v):45s} {diff:10.2f} {shift:12.1f}  {'FIXE' if fixed else 'MOBILE'}")


if __name__ == "__main__":
    main()
