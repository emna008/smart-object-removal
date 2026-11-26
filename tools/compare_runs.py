"""Compare deux dossiers de résultats (par exemple coco et world) vidéo par vidéo :
articles détectés (inventaire + nouveaux objets), événements et images par seconde.

    python tools/compare_runs.py outputs_coco outputs --names coco world
Affiche un tableau Markdown (à coller dans le README).
"""
import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict


def load(out_dir: str):
    rows = defaultdict(lambda: {"classes": Counter(), "events": Counter(), "fps": None})
    events_path = os.path.join(out_dir, "events.csv")
    if os.path.exists(events_path):
        with open(events_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                v = rows[r["video"]]
                v["events"][r["evenement"]] += 1
                if r["evenement"] in ("INVENTAIRE", "NOUVEL_OBJET"):
                    v["classes"][r["classe"]] += 1
    for path in glob.glob(os.path.join(out_dir, "resumes", "*.json")):
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        rows[s["video"]]["fps"] = s.get("fps_moyen_traitement")
    return rows


def fmt_classes(c: Counter) -> str:
    return ", ".join(f"{k}×{n}" if n > 1 else k for k, n in sorted(c.items())) or "—"


def fmt_events(c: Counter) -> str:
    keep = {k: n for k, n in c.items() if k != "INVENTAIRE"}
    return ", ".join(f"{k} {n}" for k, n in sorted(keep.items())) or "—"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dirs", nargs="+")
    p.add_argument("--names", nargs="+")
    args = p.parse_args()
    names = args.names or [os.path.basename(d.rstrip("/\\")) for d in args.dirs]
    runs = [load(d) for d in args.dirs]
    videos = sorted(set().union(*[set(r) for r in runs]))
    head = "| Vidéo | " + " | ".join(f"{n} : articles | {n} : événements | {n} img/s" for n in names) + " |"
    print(head)
    print("|" + "---|" * (1 + 3 * len(names)))
    for v in videos:
        cells = [v.replace(".mp4", "")]
        for r in runs:
            d = r.get(v)
            if d is None:
                cells += ["—", "—", "—"]
            else:
                cells += [fmt_classes(d["classes"]), fmt_events(d["events"]),
                          f"{d['fps']:.1f}" if d["fps"] else "—"]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
