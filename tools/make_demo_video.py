"""Monte la vidéo démonstrative à partir des vidéos annotées de outputs/videos/.

Chaque séquence est précédée d'un carton titre (3 s, texte sans accent) ; toutes les
images sont ramenées à la même taille (bandes noires si besoin), puis le tout est
encodé en H.264 avec le ffmpeg de imageio-ffmpeg pour être lisible partout.

    python tools/make_demo_video.py                       # montage par defaut -> outputs/demo_finale.mp4
    python tools/make_demo_video.py --out outputs/demo.mp4 --sequences "Titre A|outputs/videos/a.mp4" ...
"""
import argparse
import os
import subprocess
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZE = (1340, 540)      # taille des vidéos annotées (scène 960 px + panneau 380 px)
FPS = 30
TITLE_SECONDS = 3.0

DEFAULT = [
    ("Smart Object Removal - detection du retrait d'articles en vitrine|"
     "Vitrines nommees, DeepSORT articles + clients, inventaire, alertes RETIRE / DEPLACE / REMIS", None),
    ("1. Video synthetique : 3 retraits attribues au bon client, 1 deplacement, 1 remise|"
     "detecteur couleur, 2 vitrines, un passant ne declenche rien", "outputs/videos/synth_boutique_annotee.mp4"),
    ("2. Vraie vitrine de chaussures (Mixkit) : RETIRE 'client 2' avec trajectoire|"
     "YOLO-World, DeepSORT, verification visuelle, clip de preuve", "outputs/videos/web_mixkit_01_retrait_annotee.mp4"),
    ("3. Verre pris sur un comptoir puis repose : RETIRE a 5,1 s puis REMIS|"
     "YOLO-World, classe 'cup'", "outputs/videos/web_mixkit_06_retrait_annotee.mp4"),
    ("4. Grand magasin : 54 produits suivis, cliente qui passe, aucune fausse alerte|"
     "YOLOE, config_grand_magasin.yaml, produits masques par la cliente", "outputs/videos/web_pexels_01_passage_annotee.mp4"),
    ("5. Cas limites : trois videos filmees camera a la main|"
     "articles trop petits ou tenus en main, camera mobile : aucune fausse alerte, aucun retrait detecte", None),
    ("5a. Rayon d'hypermarche : l'homme prend une bouteille d'huile|"
     "les bouteilles du rayon ne sont pas detectees avant d'etre en main", "outputs/videos/web_pexels_02_retrait_annotee.mp4"),
    ("5b. Maroquinerie : portefeuille pris puis repose (format portrait)|"
     "les portefeuilles alignes ne sont detectes que tenus en main", "outputs/videos/web_pexels_03_retrait_annotee.mp4"),
    ("5c. Congelateur : sachet pris en gros plan|"
     "camera tres mobile, sachet jamais pose : pas d'article expose", "outputs/videos/web_pexels_04_retrait_annotee.mp4"),
]


def title_card(text: str, seconds: float):
    img = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    lines = text.split("|")
    y = SIZE[1] // 2 - 20 * (len(lines) - 1)
    for i, line in enumerate(lines):
        scale = 0.9 if i == 0 else 0.6
        color = (255, 255, 255) if i == 0 else (200, 200, 200)
        (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(img, line, ((SIZE[0] - w) // 2, y + i * 50), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    return [img] * int(round(seconds * FPS))


def fit(frame):
    """Ramène une image à SIZE sans la déformer (bandes noires)."""
    h, w = frame.shape[:2]
    scale = min(SIZE[0] / w, SIZE[1] / h)
    resized = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    y0 = (SIZE[1] - resized.shape[0]) // 2
    x0 = (SIZE[0] - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return canvas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "demo_finale.mp4"))
    p.add_argument("--sequences", nargs="*", help='"Titre|sous-titre|chemin" (chemin vide = carton seul)')
    args = p.parse_args()
    seqs = DEFAULT
    if args.sequences:
        seqs = []
        for s in args.sequences:
            parts = s.split("|")
            seqs.append(("|".join(parts[:-1]), parts[-1] or None))

    tmp = os.path.join(tempfile.gettempdir(), "demo_brut.mp4")
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    total = 0
    for title, path in seqs:
        for img in title_card(title, TITLE_SECONDS):
            writer.write(img)
            total += 1
        if not path:
            continue
        full = path if os.path.isabs(path) else os.path.join(ROOT, path)
        cap = cv2.VideoCapture(full)
        if not cap.isOpened():
            raise SystemExit(f"Video introuvable : {full}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        step = src_fps / FPS
        idx = 0.0
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if n < idx:
                continue
            idx += step
            writer.write(fit(frame))
            total += 1
        cap.release()
        print(f"  {os.path.basename(full)} ajoutee")
    writer.release()

    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "23", "-pix_fmt", "yuv420p", args.out], check=True)
    os.remove(tmp)
    print(f"{total / FPS:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
