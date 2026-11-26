"""Vidéo synthétique d'une boutique vue d'en haut (OpenCV), avec vérité terrain.

Deux vitrines contenant chacune 3 articles de couleur et de forme distinctes, et
des clients (silhouettes avec un bras) qui passent, prennent, déplacent ou
remettent des articles. 40 s, 30 img/s, 960 px de large.

Scénarios, dans l'ordre :
  0 - 4 s    scène vide et immobile (inventaire) ;
  4.5 - 10 s un client (A) passe devant la vitrine 1, bras devant un article 1 s, ne prend rien ;
  10 - 16 s  un client (B) prend un article de la vitrine 2 et sort de l'image ;
  16 - 22 s  un client (C) déplace un article à l'intérieur de la vitrine 1 ;
  22 - 32 s  un client (D) prend un article de la vitrine 1 et le remet 6 s plus tard ;
  32 - 40 s  deux clients en même temps (E passe sans rien prendre, F prend un article de la vitrine 2).

Écrit :
  videos/synth_boutique.mp4
  videos/synth_boutique.zones.json   (les deux vitrines)
  data/annotations_synth.csv         (vérité terrain exacte, avec vitrine et client)

    python tools/make_shop_video.py [--out videos/synth_boutique.mp4]

La vidéo se détecte avec `detector.type: simulated` (detectors/color_detector.py),
car YOLO ne reconnaît pas des dessins.
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from detectors.color_detector import item_color, person_color  # noqa: E402

W, H, FPS, DURATION = 960, 540, 30, 40.0
ITEM = 36                      # côté d'un article (px)
HAND_R = 26                    # rayon de la main : couvre entièrement un article
BODY = (62, 44)                # ellipse du corps (largeur, hauteur)

VITRINES = {
    "Vitrine 1": (80, 50, 450, 230),
    "Vitrine 2": (510, 50, 880, 230),
}
# nom, vitrine, forme, position de repos (centre)
ITEMS = [
    ("chaussure", "Vitrine 1", "rect", (150, 140)),
    ("sac", "Vitrine 1", "circle", (265, 140)),
    ("montre", "Vitrine 1", "triangle", (380, 140)),
    ("parfum", "Vitrine 2", "rect", (580, 140)),
    ("lunettes", "Vitrine 2", "circle", (695, 140)),
    ("bijou", "Vitrine 2", "triangle", (810, 140)),
]
AISLE_Y = 275                  # ordonnée du corps d'un client devant une vitrine


def lerp(a, b, u):
    u = max(0.0, min(1.0, u))
    return a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u


def piecewise(keys, t):
    """keys : liste (t, (x, y)) ; interpolation linéaire, None hors de l'intervalle."""
    if t < keys[0][0] or t > keys[-1][0]:
        return None
    for (t0, p0), (t1, p1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            return lerp(p0, p1, (t - t0) / (t1 - t0) if t1 > t0 else 1.0)
    return keys[-1][1]


class Client:
    """Un client : trajectoire du corps et, par segments, la position de sa main."""

    def __init__(self, name, body_keys, hand_segments):
        self.name = name
        self.body_keys = body_keys
        self.hand_segments = hand_segments  # liste (t0, t1, keys) : la main suit `keys`

    def body(self, t):
        return piecewise(self.body_keys, t)

    def hand(self, t):
        for t0, t1, keys in self.hand_segments:
            if t0 <= t <= t1:
                return piecewise(keys, t)
        return None


def reach(t0, t1, body_pos, target, hold_until, back_at):
    """Main qui part du corps, atteint `target` à t1, reste jusqu'à hold_until, revient à back_at."""
    start = (body_pos[0], body_pos[1] - BODY[1] // 2)
    keys = [(t0, start), (t1, target), (hold_until, target), (back_at, start)]
    return (t0, back_at, keys)


def build_scenario():
    clients = []
    # --- A : passe devant la vitrine 1, bras devant « sac » de 6.5 à 7.5 s, ne prend rien
    a_body = [(4.5, (-60, 280)), (6.3, (265, AISLE_Y)), (7.8, (265, AISLE_Y)), (10.0, (1020, 280))]
    clients.append(Client("A", a_body, [reach(6.3, 6.5, (265, AISLE_Y), (265, 140), 7.5, 7.8)]))
    # --- B : vient du bas, prend « parfum » (vitrine 2) à 12.5 s, repart par le bas
    b_body = [(10.0, (580, 600)), (11.8, (580, AISLE_Y)), (13.0, (580, AISLE_Y)), (15.5, (580, 620))]
    clients.append(Client("B", b_body, [reach(11.8, 12.1, (580, AISLE_Y), (580, 140), 12.5, 13.0)]))
    # --- C : déplace « montre » dans la vitrine 1 (de (380,140) à (410,190)) entre 18 et 19 s
    c_body = [(16.0, (1020, 280)), (17.5, (380, AISLE_Y)), (19.6, (380, AISLE_Y)), (22.0, (1020, 280))]
    c_hand = [(17.5, (380, AISLE_Y - BODY[1] // 2)), (18.0, (380, 140)), (19.0, (410, 190)),
              (19.2, (410, 190)), (19.6, (380, AISLE_Y - BODY[1] // 2))]
    clients.append(Client("C", c_body, [(17.5, 19.6, c_hand)]))
    # --- D : prend « chaussure » (vitrine 1) à 23.5 s, s'éloigne dans l'allée, la remet à 29.5 s
    d_body = [(22.0, (-60, 280)), (23.0, (150, AISLE_Y)), (24.0, (150, AISLE_Y)), (25.5, (150, 470)),
              (28.0, (150, 470)), (29.0, (150, AISLE_Y)), (30.0, (150, AISLE_Y)), (32.0, (-60, 280))]
    clients.append(Client("D", d_body, [reach(23.0, 23.3, (150, AISLE_Y), (150, 140), 23.5, 24.0),
                                        reach(29.0, 29.3, (150, AISLE_Y), (150, 140), 29.5, 30.0)]))
    # --- E : passe devant la vitrine 2, bras devant « bijou » de 33.8 à 34.8 s, ne prend rien
    e_body = [(32.0, (1020, 280)), (33.5, (810, AISLE_Y)), (35.6, (810, AISLE_Y)), (39.5, (-60, 280))]
    clients.append(Client("E", e_body, [reach(33.5, 33.8, (810, AISLE_Y), (810, 140), 34.8, 35.2)]))
    # --- F : vient du bas, prend « lunettes » (vitrine 2) à 34.5 s, repart par le bas
    f_body = [(32.5, (695, 600)), (34.0, (695, AISLE_Y)), (35.0, (695, AISLE_Y)), (37.5, (695, 620))]
    clients.append(Client("F", f_body, [reach(34.0, 34.2, (695, AISLE_Y), (695, 140), 34.5, 35.0)]))

    # Articles : (début, fin, position ou None=porté/invisible, client) ; sinon à sa place
    item_moves = {
        "parfum": [(12.5, DURATION, None, "B")],                       # emporté
        "montre": [(18.0, 19.0, "hand:C", "C"), (19.0, DURATION, (410, 190), "C")],  # déplacé
        "chaussure": [(23.5, 29.5, None, "D")],                        # emporté puis remis
        "lunettes": [(34.5, DURATION, None, "F")],                     # emporté
    }
    truth = [
        ("Vitrine 2", "RETIRE", "parfum", 12.5, "B"),
        ("Vitrine 1", "DEPLACE", "montre", 19.0, "C"),
        ("Vitrine 1", "RETIRE", "chaussure", 23.5, "D"),
        ("Vitrine 1", "REMIS", "chaussure", 29.5, "D"),
        ("Vitrine 2", "RETIRE", "lunettes", 34.5, "F"),
    ]
    return clients, item_moves, truth


def make_background(rng):
    bg = np.full((H, W, 3), (175, 190, 205), np.uint8)
    noise = rng.integers(-12, 13, (H, W, 1), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for x in range(0, W, 80):                       # carrelage
        cv2.line(bg, (x, 0), (x, H), (150, 165, 180), 1)
    for y in range(0, H, 80):
        cv2.line(bg, (0, y), (W, y), (150, 165, 180), 1)
    for name, (x1, y1, x2, y2) in VITRINES.items():   # vitrines : fond clair, bord sombre
        cv2.rectangle(bg, (x1, y1), (x2, y2), (238, 238, 238), -1)
        cv2.rectangle(bg, (x1, y1), (x2, y2), (60, 60, 60), 4)
        cv2.putText(bg, name.upper(), (x1 + 8, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1,
                    cv2.LINE_AA)
    cv2.putText(bg, "BOUTIQUE (vue de dessus) - video synthetique", (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (90, 90, 90), 1, cv2.LINE_AA)
    return bg


def draw_item(img, shape, center, color):
    cx, cy = int(center[0]), int(center[1])
    r = ITEM // 2
    if shape == "rect":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, -1)
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), (40, 40, 40), 1)
    elif shape == "circle":
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r, (40, 40, 40), 1, cv2.LINE_AA)
    else:
        pts = np.array([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, (40, 40, 40), 1, cv2.LINE_AA)


def draw_client(img, body, hand, color):
    bx, by = int(body[0]), int(body[1])
    if hand is not None:
        hx, hy = int(hand[0]), int(hand[1])
        cv2.line(img, (bx, by - BODY[1] // 3), (hx, hy), color, 18, cv2.LINE_AA)   # bras
        cv2.circle(img, (hx, hy), HAND_R, color, -1, cv2.LINE_AA)                 # main
    cv2.ellipse(img, (bx, by), (BODY[0] // 2, BODY[1] // 2), 0, 0, 360, color, -1, cv2.LINE_AA)
    cv2.circle(img, (bx, by), 13, color, -1, cv2.LINE_AA)                          # tête
    cv2.circle(img, (bx, by), 13, (60, 40, 20), 2, cv2.LINE_AA)


def item_position(name, t, item_moves, clients, home):
    for t0, t1, pos, who in item_moves.get(name, []):
        if t0 <= t < t1:
            if pos is None:
                return None
            if isinstance(pos, str) and pos.startswith("hand:"):
                c = next(c for c in clients if c.name == pos[5:])
                return c.hand(t) or home
            return pos
    return home


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(ROOT, "videos", "synth_boutique.mp4"))
    p.add_argument("--annotations", default=os.path.join(ROOT, "data", "annotations_synth.csv"))
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    clients, item_moves, truth = build_scenario()
    bg = make_background(rng)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = int(DURATION * FPS)
    colors = {name: item_color(name) for name, *_ in ITEMS}
    pcolor = person_color()
    for i in range(n):
        t = i / FPS
        img = bg.copy()
        for name, _, shape, home in ITEMS:
            pos = item_position(name, t, item_moves, clients, home)
            if pos is not None:
                draw_item(img, shape, pos, colors[name])
        for c in clients:
            body = c.body(t)
            if body is not None:
                draw_client(img, body, c.hand(t), pcolor)
        noise = rng.integers(-3, 4, (H, W, 1), dtype=np.int16)      # léger bruit de capteur
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(img)
    writer.release()

    zones_path = os.path.splitext(args.out)[0] + ".zones.json"
    zones = {"frame_width": W, "zones": [
        {"nom": name, "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]} for name, (x1, y1, x2, y2) in VITRINES.items()]}
    with open(zones_path, "w", encoding="utf-8") as f:
        json.dump(zones, f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(args.annotations), exist_ok=True)
    video_name = os.path.basename(args.out)
    with open(args.annotations, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "vitrine", "evenement", "classe", "temps_s", "client"])
        for zone, ev, cls, ts, who in truth:
            w.writerow([video_name, zone, ev, cls, f"{ts:.1f}", who])
    print(f"{args.out} : {n} images, {DURATION:.0f} s a {FPS} img/s")
    print(f"{zones_path} : {len(zones['zones'])} vitrines")
    print(f"{args.annotations} : {len(truth)} evenements")


if __name__ == "__main__":
    main()
