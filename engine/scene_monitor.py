"""Moniteur de scène : la logique propre au projet.

Inspiré de la « mémoire de référence » (BaselineMemory / SceneMonitor) du dépôt
kaustuv123/Missing-Object-Detection, avec les règles de notre cadrage :

1. Inventaire : pendant les premières secondes, on mémorise les articles posés
   dans chaque vitrine. Ce sont les objets de référence ; chacun appartient à la
   vitrine qui contient le centre de sa boîte.
2. Pour chaque objet de référence, à chaque image :
   - vu dans sa vitrine (même identifiant, ou nouvel identifiant au même endroit)
     -> PRESENT ; s'il s'est arrêté loin de sa place -> événement DEPLACE ;
   - pas vu mais une personne le recouvre -> MASQUE (le chronomètre s'arrête) ;
   - pas vu et rien ne le cache -> ABSENT (le chronomètre tourne) ;
   - ABSENT depuis plus de `removal_delay_seconds` et l'emplacement a changé
     visuellement -> RETIRE (alerte). Sortir de sa vitrine compte comme absent.
3. Un objet RETIRE qui revient à sa place -> REMIS.
4. Un objet nouveau, immobile pendant `new_object_seconds` -> NOUVEL_OBJET
   (il rejoint l'inventaire de sa vitrine).
5. Qui a pris l'article : quand un objet cesse d'être vu, on retient la personne
   suivie qui le recouvre ou qui est la plus proche (distance du centre de
   l'article au bord de la boîte de la personne). Son identifiant et sa
   trajectoire accompagnent l'alerte RETIRE.

Le temps utilisé est celui de la vidéo (numéro d'image / images par seconde),
ce qui rend les résultats reproductibles pour l'évaluation.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from engine.visual_check import crop_patch, patch_similarity
from engine.zone import Zone, ZoneSet
from utils.geometry import box_center, box_diag, coverage, distance, iou
from utils.types import BBox, Detection, Track

# États d'un objet de l'inventaire
PRESENT = "PRESENT"
MASQUE = "MASQUE"
ABSENT = "ABSENT"
RETIRE = "RETIRE"

# Types d'événements écrits dans le journal
EV_INVENTAIRE = "INVENTAIRE"
EV_NOUVEL_OBJET = "NOUVEL_OBJET"
EV_DEPLACE = "DEPLACE"
EV_RETIRE = "RETIRE"
EV_REMIS = "REMIS"

Point = Tuple[float, float, float]  # (temps, x, y)


@dataclass
class Event:
    time_s: float
    frame: int
    type: str
    ref_id: int
    class_name: str
    track_id: Optional[int]
    bbox: Optional[BBox]
    lost_at_s: Optional[float] = None      # instant où l'objet a cessé d'être vu
    person_near: Optional[bool] = None     # une personne était-elle à côté ?
    details: str = ""
    capture: str = ""                      # nom de la capture d'écran (rempli par le notifier)
    zone_name: str = ""                    # vitrine de l'objet
    person_id: Optional[int] = None        # client associé (RETIRE)
    clip: str = ""                         # nom du clip vidéo (rempli par le notifier)


@dataclass
class RefObject:
    """Un objet de l'inventaire d'une vitrine."""
    ref_id: int
    class_name: str
    track_id: int
    home_bbox: BBox                 # sa place « officielle »
    last_bbox: BBox                 # dernière position vue
    registered_t: float
    last_seen_t: float
    zone_name: str = ""
    state: str = PRESENT
    absent_clock: float = 0.0       # temps d'absence NON cachée (le chronomètre)
    missing_since: Optional[float] = None
    person_near_at_loss: bool = False
    taker_id: Optional[int] = None  # client associé à la disparition
    moving: bool = False
    removed_t: Optional[float] = None
    patch: Optional[np.ndarray] = None
    patch_t: float = -math.inf
    trajectory: Deque[Point] = field(default_factory=lambda: deque(maxlen=300))
    return_pts: Deque[Point] = field(default_factory=lambda: deque(maxlen=60))
    matched: Optional[Track] = None


@dataclass
class Candidate:
    """Un objet vu dans une vitrine mais pas encore dans l'inventaire."""
    track_id: int
    class_name: str
    first_t: float
    last_t: float
    bbox: BBox
    zone_name: str = ""
    centers: Deque[Point] = field(default_factory=lambda: deque(maxlen=300))


@dataclass
class Person:
    """Un client suivi par le tracker des personnes."""
    person_id: int
    bbox: BBox
    first_t: float
    last_t: float
    trajectory: Deque[Point] = field(default_factory=lambda: deque(maxlen=900))


class SceneMonitor:
    def __init__(self, config: Optional[dict] = None, zone: Optional[Union[Zone, ZoneSet]] = None):
        c = config or {}
        self.warmup_s = float(c.get("warmup_seconds", 2.0))
        self.removal_delay_s = float(c.get("removal_delay_seconds", 3.0))
        self.max_occlusion_s = float(c.get("max_occlusion_seconds", 15.0))
        self.new_object_s = float(c.get("new_object_seconds", 2.0))
        self.settle_s = float(c.get("settle_seconds", 0.7))
        self.move_ratio = float(c.get("move_ratio", 0.5))
        self.rematch_iou = float(c.get("rematch_iou", 0.3))
        self.rematch_dist_ratio = float(c.get("rematch_distance_ratio", 1.0))
        # Distance maximale (en diagonales de l'article) pour reconnaître dans un objet qui vient de se
        # poser un article manquant de la même vitrine ; null = sans limite. Sur un rayon dense de
        # produits identiques, une limite évite de « déplacer » un article vers n'importe quel voisin.
        rb = c.get("rebind_max_distance_ratio")
        self.rebind_max_dist_ratio = math.inf if rb is None else float(rb)
        self.occluder_overlap = float(c.get("occluder_overlap", 0.2))
        self.occluder_margin = float(c.get("occluder_margin", 0.5))
        self.lost_grace_s = float(c.get("lost_grace_seconds", 0.5))
        self.visual_check = bool(c.get("visual_check", True))
        self.visual_similarity = float(c.get("visual_similarity", 0.75))
        self.patch_refresh_s = float(c.get("patch_refresh_seconds", 1.0))
        self.person_memory_s = float(c.get("person_memory_seconds", 1.0))
        self.person_max_dist_ratio = float(c.get("person_max_distance_ratio", 4.0))
        self.person_forget_s = float(c.get("person_forget_seconds", 10.0))
        self.candidate_person_coverage = float(c.get("candidate_person_coverage", 0.6))
        self.duplicate_iou = float(c.get("duplicate_iou", 0.7))
        if isinstance(zone, Zone):
            zone = ZoneSet([zone])
        self.zones: ZoneSet = zone or ZoneSet()
        self.reset()

    @property
    def zone(self) -> ZoneSet:
        """Alias conservé pour l'ancien code (une seule zone)."""
        return self.zones

    # ------------------------------------------------------------------ état
    def reset(self) -> None:
        """Oublie tout et recommence la mémorisation de l'inventaire."""
        self.refs: List[RefObject] = []
        self.candidates: Dict[int, Candidate] = {}
        self.persons: Dict[int, Person] = {}
        self.start_t: Optional[float] = None
        self.prev_t: Optional[float] = None
        self.warmed_up = self.warmup_s <= 0
        self.next_ref_id = 1
        self.history: List[Event] = []
        self.stats = {"id_rebinds": 0, "visual_saves": 0, "occlusions": 0}

    def warmup_progress(self, t: float) -> float:
        if self.warmed_up or self.start_t is None or self.warmup_s <= 0:
            return 1.0
        return min(1.0, (t - self.start_t) / self.warmup_s)

    def refs_in_zone(self, zone_name: str) -> List[RefObject]:
        return [r for r in self.refs if r.zone_name == zone_name]

    # ------------------------------------------------------------ boucle
    def update(self, t: float, frame_idx: int, tracks: Sequence[Track],
               occluders: Sequence[Union[Detection, Track]] = (),
               frame: Optional[np.ndarray] = None) -> List[Event]:
        """Traite une image. `tracks` : articles suivis ; `occluders` : personnes
        (détections, ou pistes du tracker des personnes : alors on mémorise qui est où)."""
        if self.start_t is None:
            self.start_t = t
        dt = 0.0 if self.prev_t is None else max(0.0, t - self.prev_t)
        self.prev_t = t
        self._frame_idx = frame_idx
        occ_boxes = [o.bbox for o in occluders]
        self._update_persons(t, occluders)

        # Seuls les articles dont le centre est dans une vitrine comptent
        pool: Dict[int, Track] = {}
        pool_zone: Dict[int, str] = {}
        for tr in tracks:
            z = self.zones.find(tr.bbox)
            if z is not None:
                pool[tr.track_id] = tr
                pool_zone[tr.track_id] = z.name

        # 1) Association par identifiant, dans la même vitrine
        for ref in self.refs:
            ref.matched = None
        for ref in self.refs:
            if ref.state == RETIRE:
                continue
            tr = pool.get(ref.track_id)
            if tr is not None and tr.class_name == ref.class_name and pool_zone[tr.track_id] == ref.zone_name:
                ref.matched = tr
                del pool[tr.track_id]

        # 2) Ré-association par position : le tracker a changé l'identifiant,
        #    ou un objet retiré revient à sa place
        unmatched = [r for r in self.refs if r.matched is None]
        unmatched.sort(key=lambda r: r.state == RETIRE)  # les objets présents d'abord
        for ref in unmatched:
            best = self._best_rematch(ref, pool, pool_zone)
            if best is None:
                continue
            if ref.track_id != best.track_id and ref.state != RETIRE:
                self.stats["id_rebinds"] += 1
            ref.track_id = best.track_id
            ref.matched = best
            del pool[best.track_id]

        # 3) Mise à jour de chaque objet de l'inventaire
        events: List[Event] = []
        for ref in self.refs:
            events += self._update_ref(ref, t, dt, occ_boxes, frame)

        # 4) Objets restants : inventaire initial ou nouveaux objets
        events += self._update_candidates(t, pool, pool_zone, frame, occ_boxes)

        self.history.extend(events)
        return events

    # ------------------------------------------------------------ personnes
    def _update_persons(self, t: float, occluders: Sequence) -> None:
        for o in occluders:
            pid = getattr(o, "track_id", None)
            if pid is None:
                continue
            p = self.persons.get(pid)
            if p is None:
                p = Person(pid, o.bbox, t, t)
                self.persons[pid] = p
            p.bbox = o.bbox
            p.last_t = t
            p.trajectory.append((t, *box_center(o.bbox)))
        for pid in [k for k, p in self.persons.items() if t - p.last_t > self.person_forget_s]:
            del self.persons[pid]

    def visible_persons(self, t: float) -> List[Person]:
        return [p for p in self.persons.values() if p.last_t == t]

    @staticmethod
    def _edge_distance(center: Tuple[float, float], box: BBox) -> float:
        """Distance du point au bord de la boîte (0 si le point est dedans)."""
        dx = max(box[0] - center[0], 0.0, center[0] - box[2])
        dy = max(box[1] - center[1], 0.0, center[1] - box[3])
        return math.hypot(dx, dy)

    def _nearest_person(self, bbox: BBox, t: float) -> Optional[int]:
        """Le client qui recouvre l'article, sinon le plus proche (vu récemment)."""
        center = box_center(bbox)
        limit = self.person_max_dist_ratio * box_diag(bbox)
        best, best_d = None, math.inf
        for p in self.persons.values():
            if t - p.last_t > self.person_memory_s:
                continue
            d = self._edge_distance(center, p.bbox)
            if coverage(bbox, p.bbox) >= self.occluder_overlap:
                d = -coverage(bbox, p.bbox)  # recouvre l'objet : prioritaire
            if d < best_d and d <= limit:
                best, best_d = p.person_id, d
        return best

    # ------------------------------------------------------- un objet connu
    def _update_ref(self, ref: RefObject, t: float, dt: float,
                    occ_boxes: List[BBox], frame: Optional[np.ndarray]) -> List[Event]:
        events: List[Event] = []
        tr = ref.matched

        if tr is not None:
            cx, cy = box_center(tr.bbox)

            if ref.state == RETIRE:
                # L'objet revient : on attend qu'il soit posé (immobile)
                ref.return_pts.append((t, cx, cy))
                ref.last_bbox = tr.bbox
                if not self._is_stationary(ref.return_pts, t, box_diag(tr.bbox)):
                    return events
                away = t - (ref.removed_t or t)
                events.append(self._event(EV_REMIS, ref, t, tr, person_id=ref.taker_id,
                                          details=f"de retour apres {away:.1f} s"))
                ref.home_bbox = tr.bbox
                ref.trajectory.clear()
                ref.return_pts.clear()
                ref.patch = None
                ref.patch_t = -math.inf
                ref.removed_t = None

            ref.state = PRESENT
            ref.last_seen_t = t
            ref.absent_clock = 0.0
            ref.missing_since = None
            ref.taker_id = None
            ref.last_bbox = tr.bbox
            ref.trajectory.append((t, cx, cy))

            # Déplacement : signalé quand l'objet s'arrête loin de sa place
            home_diag = box_diag(ref.home_bbox)
            disp = distance((cx, cy), box_center(ref.home_bbox))
            if not ref.moving and disp > self.move_ratio * home_diag:
                ref.moving = True
            if ref.moving and self._is_stationary(ref.trajectory, t, box_diag(tr.bbox)):
                ref.moving = False
                if disp > self.move_ratio * home_diag:
                    events.append(self._event(EV_DEPLACE, ref, t, tr,
                                              details=f"deplace de {disp:.0f} px"))
                    ref.home_bbox = tr.bbox
                    ref.patch = None
                    ref.patch_t = -math.inf

            # Photo de référence de l'emplacement, quand l'objet est bien visible
            if (frame is not None and not ref.moving and not self._occluded(tr.bbox, occ_boxes)
                    and t - ref.patch_t >= self.patch_refresh_s):
                ref.patch = crop_patch(frame, tr.bbox)
                ref.patch_t = t
            return events

        # --- l'objet n'est pas vu sur cette image ---
        if ref.state == RETIRE:
            ref.return_pts.clear()
            return events

        if ref.missing_since is None:
            ref.missing_since = t
            ref.person_near_at_loss = self._person_near(ref.last_bbox, occ_boxes)
            ref.taker_id = self._nearest_person(ref.last_bbox, t)

        occluded = self._occluded(ref.last_bbox, occ_boxes)
        missing_for = t - ref.last_seen_t
        if occluded and missing_for < self.max_occlusion_s:
            if ref.state != MASQUE:
                self.stats["occlusions"] += 1
            ref.state = MASQUE              # caché : le chronomètre s'arrête
            # la personne qui recouvre l'emplacement maintenant est la plus probable
            covering = self._nearest_person(ref.last_bbox, t)
            if covering is not None:
                ref.taker_id = covering
        else:
            ref.state = ABSENT
            ref.absent_clock += dt          # visible et vide : le chronomètre tourne
            if ref.taker_id is None:
                ref.taker_id = self._nearest_person(ref.last_bbox, t)

        if ref.absent_clock < self.removal_delay_s:
            return events

        # Délai dépassé : vérification visuelle avant d'alerter
        similarity = None
        if self.visual_check and frame is not None and ref.patch is not None and not occluded:
            similarity = patch_similarity(ref.patch, crop_patch(frame, ref.last_bbox))
            if similarity is not None and similarity >= self.visual_similarity:
                # L'objet est toujours visible : le détecteur l'a raté
                self.stats["visual_saves"] += 1
                ref.state = PRESENT
                ref.absent_clock = 0.0
                ref.missing_since = None
                ref.taker_id = None
                ref.last_seen_t = t
                return events

        ref.state = RETIRE
        ref.removed_t = t
        ref.moving = False
        details = f"absent depuis {missing_for:.1f} s"
        if similarity is not None:
            details += f", similarite visuelle {similarity:.2f}"
        if ref.taker_id is not None:
            details += f", client {ref.taker_id}"
        events.append(self._event(EV_RETIRE, ref, t, None, bbox=ref.last_bbox,
                                  lost_at=ref.missing_since, person_near=ref.person_near_at_loss,
                                  person_id=ref.taker_id, details=details))
        return events

    # -------------------------------------------- inventaire / nouveaux objets
    def _update_candidates(self, t: float, pool: Dict[int, Track], pool_zone: Dict[int, str],
                           frame: Optional[np.ndarray], occ_boxes: Sequence[BBox] = ()) -> List[Event]:
        events: List[Event] = []
        for tid, tr in pool.items():
            if self._worn_or_held(tr.bbox, occ_boxes):
                continue  # porté ou tenu par une personne (lunettes, t-shirt, article en main) : pas un article exposé
            cand = self.candidates.get(tid)
            zone_name = pool_zone[tid]
            if cand is None or cand.class_name != tr.class_name or cand.zone_name != zone_name:
                cand = Candidate(tid, tr.class_name, first_t=t, last_t=t, bbox=tr.bbox, zone_name=zone_name)
                self.candidates[tid] = cand
            cand.last_t = t
            cand.bbox = tr.bbox
            cand.centers.append((t, *box_center(tr.bbox)))

        for tid in [k for k, c in self.candidates.items() if t - c.last_t > self.lost_grace_s]:
            del self.candidates[tid]

        if not self.warmed_up:
            if t - self.start_t >= self.warmup_s:
                min_presence = min(self.new_object_s, 0.5 * self.warmup_s)
                kept = [c for c in self.candidates.values()
                        if c.last_t == t and t - c.first_t >= min_presence]
                for cand in kept:
                    events.append(self._register(cand, t, frame, EV_INVENTAIRE))
                self.warmed_up = True
            return events

        for cand in list(self.candidates.values()):
            if cand.last_t != t:
                continue
            stationary = self._is_stationary(cand.centers, t, box_diag(cand.bbox))
            # Un article de la même classe manque dans cette vitrine et celui-ci vient
            # de se poser : c'est l'article déplacé (même loin de sa place), pas un nouveau
            moved = self._missing_ref_for(cand) if stationary else None
            if moved is not None:
                ev = self._rebind_moved(moved, cand, t, frame)
                if ev is not None:
                    events.append(ev)
                continue
            if t - cand.first_t < self.new_object_s or not stationary:
                continue  # un objet transporté n'est pas « posé »
            if self._duplicates_ref(cand):
                continue
            events.append(self._register(cand, t, frame, EV_NOUVEL_OBJET))
        return events

    def _missing_ref_for(self, cand: Candidate) -> Optional[RefObject]:
        """L'objet de l'inventaire (même classe, même vitrine) actuellement non vu, le plus proche."""
        best, best_d = None, math.inf
        for ref in self.refs:
            if ref.state not in (ABSENT, MASQUE) or ref.class_name != cand.class_name                     or ref.zone_name != cand.zone_name or ref.matched is not None:
                continue
            d = distance(box_center(cand.bbox), box_center(ref.last_bbox))
            if d > self.rebind_max_dist_ratio * box_diag(ref.home_bbox):
                continue  # trop loin pour être le même article
            if d < best_d:
                best, best_d = ref, d
        return best

    def _rebind_moved(self, ref: RefObject, cand: Candidate, t: float, frame: Optional[np.ndarray]) -> Optional[Event]:
        tr = Track(cand.track_id, cand.bbox, cand.class_name)
        if ref.track_id != cand.track_id:
            self.stats["id_rebinds"] += 1
        disp = distance(box_center(cand.bbox), box_center(ref.home_bbox))
        ref.track_id = cand.track_id
        ref.matched = tr
        ref.state = PRESENT
        ref.last_seen_t = t
        ref.absent_clock = 0.0
        ref.missing_since = None
        ref.taker_id = None
        ref.moving = False
        ref.last_bbox = cand.bbox
        if disp > self.move_ratio * box_diag(ref.home_bbox):
            ref.home_bbox = cand.bbox
        ref.trajectory.clear()
        ref.trajectory.append((t, *box_center(cand.bbox)))
        ref.patch = crop_patch(frame, cand.bbox) if frame is not None else None
        ref.patch_t = t if frame is not None else -math.inf
        self.candidates.pop(cand.track_id, None)
        if disp <= self.move_ratio * box_diag(ref.home_bbox):
            return None  # reposé à sa place : simple récupération d'identifiant
        return self._event(EV_DEPLACE, ref, t, tr, details=f"deplace de {disp:.0f} px (repose)")

    def _register(self, cand: Candidate, t: float, frame: Optional[np.ndarray], ev_type: str) -> Event:
        ref = RefObject(ref_id=self.next_ref_id, class_name=cand.class_name, track_id=cand.track_id,
                        home_bbox=cand.bbox, last_bbox=cand.bbox, registered_t=t, last_seen_t=t,
                        zone_name=cand.zone_name)
        ref.trajectory.append((t, *box_center(cand.bbox)))
        if frame is not None:
            ref.patch = crop_patch(frame, cand.bbox)
            ref.patch_t = t
        self.next_ref_id += 1
        self.refs.append(ref)
        self.candidates.pop(cand.track_id, None)
        details = "inventaire initial" if ev_type == EV_INVENTAIRE else "objet pose dans la vitrine"
        return Event(time_s=t, frame=self._frame_idx, type=ev_type, ref_id=ref.ref_id,
                     class_name=ref.class_name, track_id=ref.track_id, bbox=cand.bbox, details=details,
                     zone_name=ref.zone_name)

    # ------------------------------------------------------------- outils
    def _best_rematch(self, ref: RefObject, pool: Dict[int, Track],
                      pool_zone: Dict[int, str]) -> Optional[Track]:
        best, best_score = None, -math.inf
        for tr in pool.values():
            if tr.class_name != ref.class_name or pool_zone[tr.track_id] != ref.zone_name:
                continue
            # près de sa dernière position, ou de retour à sa place officielle
            for anchor in (ref.last_bbox, ref.home_bbox):
                diag = box_diag(anchor)
                overlap = iou(tr.bbox, anchor)
                d = distance(box_center(tr.bbox), box_center(anchor))
                if overlap >= self.rematch_iou or d <= self.rematch_dist_ratio * diag:
                    score = overlap - d / (10.0 * diag)
                    if score > best_score:
                        best, best_score = tr, score
        return best

    def _worn_or_held(self, bbox: BBox, occ_boxes: Sequence[BBox]) -> bool:
        """Boîte en grande partie à l'intérieur de celle d'une personne."""
        return any(coverage(bbox, ob) >= self.candidate_person_coverage for ob in occ_boxes)

    def _duplicates_ref(self, cand: Candidate) -> bool:
        """Le candidat est-il déjà un article de l'inventaire ? Même classe et recouvrement > 0,5,
        ou recouvrement > `duplicate_iou` quelle que soit la classe : un détecteur à vocabulaire ouvert
        nomme le même vêtement tantôt « dress », tantôt « shirt », ce n'est pas un nouvel article."""
        for r in self.refs:
            if r.state == RETIRE:
                continue
            overlap = iou(r.last_bbox, cand.bbox)
            if (r.class_name == cand.class_name and overlap > 0.5) or overlap >= self.duplicate_iou:
                return True
        return False

    def _occluded(self, bbox: BBox, occ_boxes: List[BBox]) -> bool:
        """Caché si une personne recouvre l'objet, ou le touche presque
        (une main qui manipule l'objet est souvent juste à côté de sa boîte)."""
        margin = self.occluder_margin * box_diag(bbox)
        center = box_center(bbox)
        for ob in occ_boxes:
            if coverage(bbox, ob) >= self.occluder_overlap:
                return True
            if margin > 0 and self._edge_distance(center, ob) <= margin:
                return True
        return False

    def _person_near(self, bbox: BBox, occ_boxes: List[BBox]) -> bool:
        diag = box_diag(bbox)
        c = box_center(bbox)
        for ob in occ_boxes:
            if coverage(bbox, ob) > 0 or self._edge_distance(c, ob) <= 1.5 * diag:
                return True
        return False

    def _is_stationary(self, pts: Sequence[Point], t: float, diag: float) -> bool:
        """Immobile si, sur les `settle_seconds` dernières secondes, le centre
        a bougé de moins de 10 % de la diagonale de la boîte."""
        window = [p for p in pts if p[0] >= t - self.settle_s - 1e-9]
        if len(window) < 2 or window[-1][0] - window[0][0] < 0.8 * self.settle_s:
            return False
        lx, ly = window[-1][1], window[-1][2]
        return max(math.hypot(x - lx, y - ly) for _, x, y in window) <= 0.1 * diag

    def _event(self, ev_type: str, ref: RefObject, t: float, tr: Optional[Track], bbox: Optional[BBox] = None,
               lost_at: Optional[float] = None, person_near: Optional[bool] = None,
               person_id: Optional[int] = None, details: str = "") -> Event:
        return Event(time_s=t, frame=self._frame_idx, type=ev_type, ref_id=ref.ref_id,
                     class_name=ref.class_name, track_id=tr.track_id if tr else ref.track_id,
                     bbox=bbox if bbox is not None else (tr.bbox if tr else ref.last_bbox),
                     lost_at_s=lost_at, person_near=person_near, details=details,
                     zone_name=ref.zone_name, person_id=person_id)
