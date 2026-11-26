"""Lance le code du dépôt de référence (kaustuv123/Missing-Object-Detection) sur une
vidéo, sans le modifier, et écrit ses alertes dans un CSV comparable au nôtre.

Son moteur relit config.yaml et ignore l'option --video : on lui passe une copie de
sa config (dans outputs/) où seul le chemin de la vidéo est changé, sans modifier le dépôt.
Le dépôt n'a ni journal ni option sans fenêtre : on remplace cv2.imshow / waitKey
par des fonctions neutres, on intercepte ses messages console (« Object ID n is now
considered MISSING », « NEW object detected with ID n ») et on y ajoute le numéro
d'image courant pour obtenir un temps vidéo.

    git clone https://github.com/kaustuv123/Missing-Object-Detection reference/Missing-Object-Detection
    python tools/run_reference.py --video videos/ref_kaustuv_bouteille.mp4
Résultat : outputs/reference_events.csv (video, temps_s, image, evenement, objet).
"""
import argparse
import builtins
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "reference", "Missing-Object-Detection")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default=os.path.join(ROOT, "videos", "ref_kaustuv_bouteille.mp4"))
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "reference_events.csv"))
    args = p.parse_args()
    if not os.path.isdir(REF):
        raise SystemExit(f"Depot de reference introuvable : {REF}")

    import cv2
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: -1
    cv2.destroyAllWindows = lambda *a, **k: None
    cv2.namedWindow = lambda *a, **k: None

    sys.path.insert(0, REF)
    os.chdir(REF)                      # ses chemins (config.yaml, yolov8n.pt) sont relatifs
    import main as ref_main            # noqa: E402  (le main.py du depot)
    from engine.inference import InferenceEngine  # noqa: E402

    state = {"frame": 0}
    original_process = InferenceEngine._process_frame

    def counted(self, frame):
        state["frame"] = self.frame_count
        return original_process(self, frame)

    InferenceEngine._process_frame = counted

    rows = []
    patterns = [(re.compile(r"Object ID (\d+) is now considered MISSING"), "MISSING"),
                (re.compile(r"NEW object detected with ID (\d+)"), "NEW"),
                (re.compile(r"Object ID (\d+) has reappeared"), "REAPPEARED"),
                (re.compile(r"Baseline established with (\d+) objects"), "BASELINE")]
    real_print = builtins.print

    def spy_print(*a, **k):
        text = " ".join(str(x) for x in a)
        for rx, kind in patterns:
            m = rx.search(text)
            if m:
                rows.append([os.path.basename(args.video), f"{state['frame'] / fps:.2f}", state["frame"], kind, m.group(1)])
        real_print(*a, **k)

    builtins.print = spy_print
    import yaml
    with open(os.path.join(REF, "config.yaml"), "r", encoding="utf-8") as f:
        ref_cfg = yaml.safe_load(f)
    ref_cfg["video"]["source"] = os.path.abspath(args.video)
    cfg_copy = os.path.abspath(os.path.join(os.path.dirname(args.out), "reference_config.yaml"))
    os.makedirs(os.path.dirname(cfg_copy), exist_ok=True)
    with open(cfg_copy, "w", encoding="utf-8") as f:
        yaml.safe_dump(ref_cfg, f)
    sys.argv = ["main.py", "--config", cfg_copy]
    try:
        ref_main.main()
    finally:
        builtins.print = real_print

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "temps_s", "image", "evenement", "objet"])
        w.writerows(rows)
    real_print(f"\n{len(rows)} evenements du depot de reference -> {args.out}")
    for r in rows:
        real_print("  ", r)


if __name__ == "__main__":
    main()
