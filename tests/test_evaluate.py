import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("evaluate", os.path.join(ROOT, "tools", "evaluate.py"))
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def test_association_alertes_et_verite():
    truths = [("cup", 10.0), ("book", 30.0)]
    alerts = [("cup", 13.2), ("cup", 50.0)]          # une juste, une fausse ; le livre est raté
    tp, fp, fn, delays = evaluate.match(truths, alerts, before=1.0, after=10.0)
    assert (tp, fp, fn) == (1, 1, 1)
    assert delays == pytest.approx([3.2])


def test_mauvaise_classe_ne_compte_pas():
    tp, fp, fn, _ = evaluate.match([("cup", 10.0)], [("bottle", 12.0)], 1.0, 10.0)
    assert (tp, fp, fn) == (0, 1, 1)


def test_vitrine_prise_en_compte():
    truths = [("Vitrine A", "shoe", 10.0)]
    assert evaluate.match(truths, [("Vitrine B", "shoe", 12.0)], 1.0, 10.0)[:3] == (0, 1, 1)
    assert evaluate.match(truths, [("Vitrine A", "shoe", 12.0)], 1.0, 10.0)[:3] == (1, 0, 0)
    # vitrine ou classe vides dans l'annotation : n'importe laquelle convient
    assert evaluate.match([("", "", 10.0)], [("Vitrine B", "bag", 12.0)], 1.0, 10.0)[:3] == (1, 0, 0)


def test_lecture_annotations_avec_et_sans_vitrine(tmp_path):
    old = tmp_path / "old.csv"
    old.write_text("video,evenement,classe,temps_s\nclip.mp4,RETIRE,cup,12.5\nclip2.mp4,AUCUN,,\n", encoding="utf-8")
    new = tmp_path / "new.csv"
    new.write_text("video,vitrine,evenement,classe,temps_s\nclip.mp4,Vitrine A,RETIRE,,8.3\n", encoding="utf-8")
    t_old = evaluate.load_truths(str(old), "RETIRE")
    assert t_old["clip.mp4"] == [("", "cup", 12.5)] and t_old["clip2.mp4"] == []
    t_new = evaluate.load_truths(str(new), "RETIRE")
    assert t_new["clip.mp4"] == [("Vitrine A", "", 8.3)]
