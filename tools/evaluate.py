"""Évaluation des alertes par rapport aux annotations (vérité terrain).

Annotations (data/annotations.csv), une ligne par retrait réel :
    video,vitrine,evenement,classe,temps_s
    web_mixkit_01_retrait.mp4,Vitrine chaussures,RETIRE,shoe,6.5
    web_mixkit_03_manipulation.mp4,,AUCUN,,          <- vidéo sans retrait (compte les fausses alertes)
La colonne `vitrine` est facultative (ancien format : video,evenement,classe,temps_s).
Quand elle est renseignée, l'alerte doit venir de la même vitrine. La classe est
aussi facultative : vide = n'importe quelle classe.

Une alerte est juste (vrai positif) si elle concerne la même vitrine et la même
classe et arrive entre `--avant` secondes avant et `--apres` secondes après
l'instant annoté.

Mesures : précision, rappel, F1, délai moyen d'alerte, fausses alertes par minute.

    python tools/evaluate.py --annotations data/annotations.csv
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def read_rows(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _norm(item) -> Tuple[str, str, float]:
    """(vitrine, classe, temps) ; accepte aussi l'ancien (classe, temps)."""
    if len(item) == 2:
        return "", item[0] or "", float(item[1])
    return item[0] or "", item[1] or "", float(item[2])


def match(truths: List[tuple], alerts: List[tuple], before: float, after: float) -> Tuple[int, int, int, List[float]]:
    """Associe chaque retrait réel à au plus une alerte. Renvoie (VP, FP, FN, délais)."""
    truths = [_norm(x) for x in truths]
    alerts = [_norm(x) for x in alerts]
    used = set()
    delays = []
    for zone, cls, t_true in sorted(truths, key=lambda x: x[2]):
        best = None
        for i, (a_zone, a_cls, t_alert) in enumerate(alerts):
            if i in used or (cls and a_cls != cls) or (zone and a_zone != zone):
                continue
            d = t_alert - t_true
            if -before <= d <= after and (best is None or abs(d) < abs(best[1])):
                best = (i, d)
        if best is not None:
            used.add(best[0])
            delays.append(best[1])
    tp = len(delays)
    return tp, len(alerts) - tp, len(truths) - tp, delays


def ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def load_truths(path: str, event: str) -> Dict[str, List[tuple]]:
    """Lit les annotations, avec ou sans colonne vitrine."""
    truths: Dict[str, List[tuple]] = defaultdict(list)
    for row in read_rows(path):
        video = (row.get("video") or "").strip()
        if not video:
            continue
        truths.setdefault(video, [])
        if (row.get("evenement") or "").strip().upper() == event and (row.get("temps_s") or "").strip():
            truths[video].append(((row.get("vitrine") or "").strip(), (row.get("classe") or "").strip(),
                                  float(row["temps_s"])))
    return truths


def load_alerts(path: str, event: str) -> Dict[str, List[tuple]]:
    alerts: Dict[str, List[tuple]] = defaultdict(list)
    for row in read_rows(path):
        if row["evenement"] == event:
            alerts[row["video"]].append(((row.get("vitrine") or "").strip(), row["classe"], float(row["temps_s"])))
    return alerts


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", default=os.path.join(root, "data", "annotations.csv"))
    p.add_argument("--events", default=os.path.join(root, "outputs", "events.csv"))
    p.add_argument("--resumes", default=os.path.join(root, "outputs", "resumes"))
    p.add_argument("--event", default="RETIRE", help="type d'evenement evalue (RETIRE, REMIS, DEPLACE...)")
    p.add_argument("--avant", type=float, default=1.0, help="tolerance avant l'instant annote (s)")
    p.add_argument("--apres", type=float, default=10.0, help="tolerance apres l'instant annote (s)")
    p.add_argument("--sortie", default=os.path.join(root, "outputs", "evaluation.csv"))
    args = p.parse_args()

    truths = load_truths(args.annotations, args.event)
    alerts = load_alerts(args.events, args.event)

    missing = [v for v in truths if not os.path.exists(os.path.join(args.resumes, os.path.splitext(v)[0] + ".json"))]
    if missing:
        print("Attention, videos annotees mais pas encore traitees :", ", ".join(missing))
    ignored = sorted(set(alerts) - set(truths))
    if ignored:
        print("Videos traitees mais non annotees (ignorees) :", ", ".join(ignored))

    lines = []
    total = {"vp": 0, "fp": 0, "fn": 0, "duree": 0.0, "delais": []}
    for video in sorted(truths):
        if video in missing:
            continue
        tp, fp, fn, delays = match(truths[video], alerts.get(video, []), args.avant, args.apres)
        with open(os.path.join(args.resumes, os.path.splitext(video)[0] + ".json"), encoding="utf-8") as f:
            duration = float(json.load(f).get("duree_video_s", 0.0))
        total["vp"] += tp
        total["fp"] += fp
        total["fn"] += fn
        total["duree"] += duration
        total["delais"] += delays
        lines.append([video, len(truths[video]), tp, fp, fn,
                      f"{sum(delays) / len(delays):.2f}" if delays else "", f"{duration:.1f}"])

    tp, fp, fn = total["vp"], total["fp"], total["fn"]
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = ratio(2 * precision * recall, precision + recall)
    mean_delay = ratio(sum(total["delais"]), len(total["delais"]))
    fp_per_min = ratio(fp, total["duree"] / 60.0)

    header = ["video", "reels", "VP", "FP", "FN", "delai_moyen_s", "duree_s"]
    widths = [max(len(str(r[i])) for r in [header] + lines) for i in range(len(header))]
    print("\n" + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for r in lines:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    print(f"\nEvenement evalue : {args.event}")
    print(f"Precision : {precision:.2f}   Rappel : {recall:.2f}   F1 : {f1:.2f}")
    print(f"Delai moyen d'alerte : {mean_delay:.2f} s   Fausses alertes : {fp_per_min:.2f} par minute")

    os.makedirs(os.path.dirname(args.sortie), exist_ok=True)
    with open(args.sortie, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(lines)
        w.writerow(["TOTAL", tp + fn, tp, fp, fn, f"{mean_delay:.2f}", f"{total['duree']:.1f}"])
        w.writerow([])
        w.writerow(["precision", f"{precision:.3f}", "rappel", f"{recall:.3f}", "f1", f"{f1:.3f}",
                    "fausses_alertes_par_min", f"{fp_per_min:.3f}"])
    print(f"Resultats enregistres dans {args.sortie}")


if __name__ == "__main__":
    main()
