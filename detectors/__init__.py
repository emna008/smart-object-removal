"""Fabrique des détecteurs : `detector.type` dans config.yaml.

- coco      : YOLOv8n préentraîné COCO (choix du dépôt de référence) ;
- world     : modèle à vocabulaire ouvert (YOLO-World / YOLOE), classes en texte ;
- simulated : détecteur par couleur, pour la vidéo synthétique et les tests.
"""

DETECTOR_TYPES = ("coco", "world", "simulated")


def build_detector(config: dict):
    kind = (config.get("type") or "coco").lower()
    if kind == "coco":
        from detectors.yolo_detector import YOLODetector
        return YOLODetector(config)
    if kind == "world":
        from detectors.world_detector import WorldDetector
        return WorldDetector(config)
    if kind == "simulated":
        from detectors.color_detector import ColorDetector
        return ColorDetector(config)
    raise ValueError(f"detector.type inconnu : '{kind}'. Choix : {', '.join(DETECTOR_TYPES)}")
