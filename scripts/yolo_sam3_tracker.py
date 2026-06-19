"""
YoloSam3Tracker — YOLO+SAM3 ensemble para tracking robusto.

Tras la propagación estándar de SAM3, ejecuta YOLO frame por frame para
detectar objetos que SAM3 perdió (score bajo o máscara vacía) y los rescata
usando las detecciones de YOLO como fuente de verdad alternativa.

Innovación demostrable:
  · Métricas antes/después: continuidad de tracks, frames perdidos vs recuperados
  · Rescue events registrados en el JSON de tracks para análisis cuantitativo
  · IoU between SAM3 masks y YOLO boxes como medida de calidad por frame
"""

import os
import cv2
import numpy as np
import torch


# Umbral de score SAM3 por debajo del cual consideramos que el objeto está "perdido"
SAM3_LOSS_SCORE = 0.30
# Umbral mínimo de píxeles en la máscara para que no se considere vacía
MASK_MIN_PIXELS = 200
# IoU mínimo entre máscara SAM3 y caja YOLO para que sean el mismo objeto
IOU_MATCH_THRESHOLD = 0.15
# Distancia máxima (píxeles) al último centroide conocido para match de YOLO
MAX_RESCUE_DIST = 300


def _mask_from_np(mask, frame_shape):
    """Convierte máscara (Tensor/ndarray/None) a ndarray bool con shape (H,W)."""
    if mask is None:
        return np.zeros(frame_shape[:2], dtype=bool)
    if isinstance(mask, torch.Tensor):
        m = mask.cpu().numpy()
    else:
        m = np.asarray(mask)
    return m.astype(bool).reshape(frame_shape[:2])


def _box_mask(box_xyxy, shape):
    """Crea máscara booleana rectangular a partir de [x1,y1,x2,y2]."""
    m = np.zeros(shape[:2], dtype=bool)
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(shape[1], x2), min(shape[0], y2)
    m[y1:y2, x1:x2] = True
    return m


def _iou_mask_box(mask_np: np.ndarray, box_xyxy: list) -> float:
    """IoU entre máscara binaria y caja bounding box (caja = región rellena)."""
    if mask_np is None:
        return 0.0
    H, W = mask_np.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    box_mask = np.zeros_like(mask_np, dtype=bool)
    box_mask[y1:y2, x1:x2] = True
    m = mask_np.astype(bool)
    inter = (m & box_mask).sum()
    union = (m | box_mask).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _centroid_from_mask(mask_np: np.ndarray):
    """Retorna (cx, cy) desde una máscara binaria, o None si está vacía."""
    if mask_np is None:
        return None
    ys, xs = np.where(mask_np)
    if len(xs) == 0:
        return None
    return (int(xs.mean()), int(ys.mean()))


def _dist(p1, p2) -> float:
    if p1 is None or p2 is None:
        return float("inf")
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


class YoloSam3Tracker:
    """
    Envuelve la salida de SAM3 y aplica rescate YOLO para objetos perdidos.

    Uso típico:
        tracker = YoloSam3Tracker(yolo_path="runs/detect/train-2/weights/best.pt")
        outputs, rescue_events, metrics = tracker.rescue(
            outputs_per_frame, frame_paths, obj_id_to_label
        )
    """

    def __init__(
        self,
        yolo_path: str = None,
        sam3_loss_score: float = SAM3_LOSS_SCORE,
        mask_min_pixels: int = MASK_MIN_PIXELS,
        iou_match_threshold: float = IOU_MATCH_THRESHOLD,
        max_rescue_dist: float = MAX_RESCUE_DIST,
        conf: float = 0.20,
    ):
        self.sam3_loss_score = sam3_loss_score
        self.mask_min_pixels = mask_min_pixels
        self.iou_match_threshold = iou_match_threshold
        self.max_rescue_dist = max_rescue_dist
        self.conf = conf
        self._yolo = None

        if yolo_path is None:
            default = os.path.join(
                os.path.dirname(__file__), "..", "runs", "detect", "train-2", "weights", "best.pt"
            )
            yolo_path = os.path.abspath(default)

        if os.path.exists(yolo_path):
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(yolo_path)
                print(f"[YoloSam3Tracker] Modelo cargado: {yolo_path}")
            except ImportError:
                print("[YoloSam3Tracker] ultralytics no disponible — sin rescate YOLO")
        else:
            print(f"[YoloSam3Tracker] Modelo no encontrado: {yolo_path}")

    def available(self) -> bool:
        return self._yolo is not None

    def detect(self, frame_bgr: np.ndarray) -> list:
        """
        Ejecuta YOLO en el frame. Retorna lista de dicts:
          {"label_class": "robot"|"ball", "cx", "cy", "box_xyxy", "conf", "area"}
        """
        if self._yolo is None:
            return []
        results = self._yolo(frame_bgr, conf=self.conf, verbose=False)
        dets = []
        for r in results:
            for box in r.boxes:
                cls_name = self._yolo.names[int(box.cls[0])]
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                dets.append({
                    "label_class": cls_name,
                    "cx": (x1 + x2) // 2,
                    "cy": (y1 + y2) // 2,
                    "box_xyxy": [x1, y1, x2, y2],
                    "conf": float(box.conf[0]),
                    "area": (x2 - x1) * (y2 - y1),
                })
        return dets

    def _is_lost(self, obj_data: dict) -> bool:
        """Determina si SAM3 perdió un objeto en este frame."""
        score = obj_data.get("score", 1.0)
        mask = obj_data.get("mask")
        if mask is None:
            return True
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(mask).astype(bool)
        pixel_count = int(mask_np.sum())
        return score < self.sam3_loss_score or pixel_count < self.mask_min_pixels

    def _label_to_class(self, label: str) -> str:
        """Mapea etiqueta de pipeline ('ball', 'robot1', 'robot2', ...) a clase YOLO."""
        return "ball" if label == "ball" else "robot"

    def _find_last_centroid(self, outputs_per_frame: dict, obj_id: int, before_fidx: int):
        """Busca el último centroide conocido del objeto antes del frame dado."""
        for f in range(before_fidx - 1, -1, -1):
            obj_data = outputs_per_frame.get(f, {}).get(obj_id)
            if obj_data and not self._is_lost(obj_data):
                mask = obj_data.get("mask")
                if mask is not None:
                    if isinstance(mask, torch.Tensor):
                        m = mask.cpu().numpy().astype(bool)
                    else:
                        m = np.asarray(mask).astype(bool)
                    c = _centroid_from_mask(m)
                    if c:
                        return c
        return None

    def _match_yolo_to_obj(
        self,
        candidates: list,
        obj_data: dict,
        last_centroid,
    ):
        """
        Selecciona la detección YOLO que mejor corresponde al objeto perdido.
        Criterios (en orden): IoU > threshold > distancia al último centroide conocido.
        """
        if not candidates:
            return None

        mask = obj_data.get("mask")
        mask_np = None
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy().astype(bool)
            else:
                mask_np = np.asarray(mask).astype(bool)

        # Sin posición conocida: tomar la detección de mayor confianza de la clase.
        # No hay forma de saber cuál de las candidatas es el objeto correcto.
        if last_centroid is None:
            return max(candidates, key=lambda d: d["conf"])

        best = None
        best_score = -1.0

        for det in candidates:
            iou = _iou_mask_box(mask_np, det["box_xyxy"]) if mask_np is not None else 0.0
            dist = _dist(last_centroid, (det["cx"], det["cy"]))
            if dist > self.max_rescue_dist:
                continue
            dist_score = 1.0 - dist / self.max_rescue_dist
            combo = iou * 0.4 + dist_score * 0.6
            if combo > best_score:
                best_score = combo
                best = det

        return best

    def relabel_by_yolo(
        self,
        outputs_per_frame: dict,
        frame_paths: list,
        obj_id_to_label: dict,
    ) -> int:
        """
        Corrige swaps de identidad entre robots idénticos.

        SAM3 (multi-sesión) puede rastrear el mismo robot físico en dos sesiones
        distintas porque el VG no distingue objetos visualmente iguales.
        Este paso usa YOLO cada frame para re-asignar qué máscara pertenece a
        qué obj_id según proximidad al centroide esperado.

        Retorna el número de frames donde se corrigió al menos un label.
        """
        if not self.available():
            return 0

        robot_oids = sorted(
            [oid for oid, lbl in obj_id_to_label.items() if lbl != "ball"]
        )
        if len(robot_oids) < 2:
            return 0   # un solo robot → no hay swap posible

        # Etiquetas canónicas por rango de área YOLO (mayor → robot1, etc.)
        canonical = {i: f"robot{i+1}" for i in range(len(robot_oids))}

        corrected_frames = 0

        for fidx in sorted(outputs_per_frame.keys()):
            frame_bgr = cv2.imread(frame_paths[fidx])
            if frame_bgr is None:
                continue

            yolo_dets  = self.detect(frame_bgr)
            robot_dets = sorted(
                [d for d in yolo_dets if d["label_class"] == "robot"],
                key=lambda d: d["area"], reverse=True,
            )
            if len(robot_dets) < len(robot_oids):
                continue   # no hay suficientes detecciones para re-etiquetar

            frame_objs = outputs_per_frame[fidx]

            # Centroide SAM3 por obj_id
            centroids = {}
            for oid in robot_oids:
                od = frame_objs.get(oid)
                if od:
                    m = _mask_from_np(od.get("mask"), frame_bgr.shape)
                    centroids[oid] = _centroid_from_mask(m)

            # Asignación greedy: cada obj_id toma el robot_det más cercano
            available = list(range(len(robot_dets)))
            assignment = {}    # oid → det_idx
            for oid in robot_oids:
                c = centroids.get(oid)
                if c is None or not available:
                    continue
                best = min(
                    available,
                    key=lambda i: _dist(c, (robot_dets[i]["cx"], robot_dets[i]["cy"])),
                )
                assignment[oid] = best
                available.remove(best)

            # Aplicar etiqueta canónica según det_idx asignado
            changed = False
            for oid, det_idx in assignment.items():
                new_label = canonical[det_idx]
                if frame_objs.get(oid, {}).get("label") != new_label:
                    frame_objs.setdefault(oid, {"mask": None, "score": 0.0})
                    frame_objs[oid]["label"] = new_label
                    changed = True

            if changed:
                corrected_frames += 1

        return corrected_frames

    def rescue(
        self,
        outputs_per_frame: dict,
        frame_paths: list,
        obj_id_to_label: dict,
    ) -> tuple:
        """
        Aplica rescate YOLO a los frames donde SAM3 perdió objetos.

        Parámetros:
            outputs_per_frame: {fidx: {obj_id: {mask, score, label}}}
            frame_paths:       lista ordenada de rutas a los frames (BGR)
            obj_id_to_label:   {obj_id: "ball"|"robot1"|...}

        Retorna:
            (outputs_per_frame_updated, rescue_events, metrics)
            · rescue_events: lista de dicts con info por frame/objeto rescatado
            · metrics: resumen cuantitativo
        """
        if not self.available():
            print("[YoloSam3Tracker] YOLO no disponible — sin rescate")
            return outputs_per_frame, [], {}

        sorted_frames = sorted(outputs_per_frame.keys())
        all_obj_ids = set(obj_id_to_label.keys())
        rescue_events = []
        total_obj_frames = 0
        lost_obj_frames = 0
        rescued_count = 0
        sam3_iou_sum = 0.0
        sam3_iou_n = 0

        # Diagnóstico: cuántos obj_ids produce SAM3 por frame
        sam3_ids_per_frame = [len(outputs_per_frame.get(f, {})) for f in sorted_frames]
        print(f"[YoloSam3Tracker] Analizando {len(sorted_frames)} frames, "
              f"{len(obj_id_to_label)} objetos esperados...")
        print(f"  SAM3 produce por frame: min={min(sam3_ids_per_frame)} "
              f"max={max(sam3_ids_per_frame)} "
              f"media={sum(sam3_ids_per_frame)/len(sam3_ids_per_frame):.1f}")
        missing_ids = all_obj_ids - set().union(*[set(outputs_per_frame.get(f, {}).keys())
                                                   for f in sorted_frames])
        if missing_ids:
            print(f"  Obj_ids ausentes en TODO el video: "
                  f"{[f'{oid}={obj_id_to_label[oid]}' for oid in sorted(missing_ids)]}")

        for fidx in sorted_frames:
            frame_objs = outputs_per_frame.setdefault(fidx, {})

            # Inyectar entrada vacía para objetos que SAM3 nunca produjo en este frame.
            # Sin esto el rescue ni los ve.
            for obj_id in all_obj_ids:
                if obj_id not in frame_objs:
                    frame_objs[obj_id] = {
                        "label": obj_id_to_label[obj_id],
                        "score": 0.0,
                        "mask":  None,
                    }

            lost_ids = []
            for obj_id, obj_data in frame_objs.items():
                if obj_id not in all_obj_ids:
                    continue  # objeto extra no esperado
                total_obj_frames += 1
                if self._is_lost(obj_data):
                    lost_ids.append(obj_id)
                    lost_obj_frames += 1
                else:
                    sam3_iou_sum += obj_data.get("score", 0.0)
                    sam3_iou_n += 1

            if not lost_ids:
                continue

            # Ejecuta YOLO en este frame
            frame_bgr = cv2.imread(frame_paths[fidx])
            if frame_bgr is None:
                continue
            yolo_dets = self.detect(frame_bgr)

            for obj_id in lost_ids:
                obj_data = frame_objs[obj_id]
                label = obj_id_to_label.get(obj_id, f"obj_{obj_id}")
                target_class = self._label_to_class(label)

                candidates = [d for d in yolo_dets if d["label_class"] == target_class]
                last_c = self._find_last_centroid(outputs_per_frame, obj_id, fidx)

                best_det = self._match_yolo_to_obj(candidates, obj_data, last_c)

                if best_det:
                    obj_data["rescued"] = True
                    obj_data["yolo_box"] = best_det["box_xyxy"]
                    obj_data["yolo_conf"] = best_det["conf"]
                    obj_data["yolo_cx"] = best_det["cx"]
                    obj_data["yolo_cy"] = best_det["cy"]
                    obj_data["centroid_rescue"] = (best_det["cx"], best_det["cy"])

                    mask = obj_data.get("mask")
                    iou_val = 0.0
                    if mask is not None:
                        if isinstance(mask, torch.Tensor):
                            m = mask.cpu().numpy().astype(bool)
                        else:
                            m = np.asarray(mask).astype(bool)
                        iou_val = _iou_mask_box(m, best_det["box_xyxy"])

                    rescue_events.append({
                        "frame_idx": fidx,
                        "obj_id": obj_id,
                        "label": label,
                        "sam3_score": obj_data.get("score", 0.0),
                        "yolo_conf": best_det["conf"],
                        "yolo_box": best_det["box_xyxy"],
                        "yolo_iou_vs_sam3": round(iou_val, 4),
                        "last_known_centroid": list(last_c) if last_c else None,
                    })
                    rescued_count += 1
                else:
                    obj_data["rescued"] = False

        # Métricas cuantitativas
        continuity_before = 1.0 - (lost_obj_frames / total_obj_frames) if total_obj_frames else 1.0
        continuity_after = 1.0 - ((lost_obj_frames - rescued_count) / total_obj_frames) if total_obj_frames else 1.0
        metrics = {
            "total_obj_frames": total_obj_frames,
            "lost_obj_frames": lost_obj_frames,
            "rescued_frames": rescued_count,
            "unrecovered_frames": lost_obj_frames - rescued_count,
            "track_continuity_sam3_only": round(continuity_before, 4),
            "track_continuity_yolo_rescued": round(continuity_after, 4),
            "continuity_improvement": round(continuity_after - continuity_before, 4),
            "mean_sam3_score_when_tracked": round(sam3_iou_sum / sam3_iou_n, 4) if sam3_iou_n else 0.0,
        }

        print(f"[YoloSam3Tracker] Rescate completo:")
        print(f"  Frames×objetos totales : {total_obj_frames}")
        print(f"  Perdidos por SAM3      : {lost_obj_frames}  ({100*continuity_before:.1f}% continuidad)")
        print(f"  Rescatados por YOLO    : {rescued_count}")
        print(f"  Sin recuperar          : {lost_obj_frames - rescued_count}")
        print(f"  Continuidad final      : {100*continuity_after:.1f}%  (+{100*(continuity_after-continuity_before):.1f}%)")

        return outputs_per_frame, rescue_events, metrics

    def fix_fusion(
        self,
        outputs_per_frame: dict,
        frame_paths: list,
        obj_id_to_label: dict,
    ) -> tuple:
        """
        Detecta y corrige frames donde dos robots se fusionaron en una sola máscara.

        Síntoma de fusión:
          - obj_id A (robot) tiene una máscara grande que cubre la posición de otro robot
          - obj_id B (robot) tiene máscara vacía o de score bajo en ese mismo frame

        Corrección (inspirada en yolo_sam2_cell_tracker._apply_corrections):
          - YOLO detecta ambos robots en el frame de fusión
          - Máscara de A se recorta a su caja YOLO
          - Máscara de B se construye como intersección de la máscara fusionada con la
            caja YOLO de B (recupera la forma exacta de B dentro del blob fusionado)

        Retorna (outputs_per_frame_corregido, fusion_events)
        """
        if not self.available():
            return outputs_per_frame, []

        # Identificar obj_ids de robots (pueden fusionarse entre sí)
        robot_ids = [oid for oid, lbl in obj_id_to_label.items() if lbl != "ball"]
        if len(robot_ids) < 2:
            return outputs_per_frame, []

        sorted_frames = sorted(outputs_per_frame.keys())
        fusion_events = []
        fixed_count = 0

        for fidx in sorted_frames:
            frame_objs = outputs_per_frame[fidx]

            # Buscar pares (robot_sano, robot_perdido)
            for lost_id in robot_ids:
                lost_data = frame_objs.get(lost_id, {})
                if not self._is_lost(lost_data):
                    continue

                # robot_id perdido → buscar cuál robot "se lo tragó"
                for donor_id in robot_ids:
                    if donor_id == lost_id:
                        continue
                    donor_data = frame_objs.get(donor_id, {})
                    if self._is_lost(donor_data):
                        continue

                    frame_shape = cv2.imread(frame_paths[fidx]).shape
                    donor_mask = _mask_from_np(donor_data.get("mask"), frame_shape)
                    donor_pixels = int(donor_mask.sum())

                    # Un robot normal ocupa ~5000-20000px; >30000 sugiere fusión
                    if donor_pixels < 30000:
                        continue

                    # Verificar con YOLO que el robot perdido está cerca del donor mask
                    frame_bgr = cv2.imread(frame_paths[fidx])
                    yolo_dets = self.detect(frame_bgr)
                    robot_dets = [d for d in yolo_dets if d["label_class"] == "robot"]

                    if len(robot_dets) < 2:
                        continue

                    # Encontrar qué detección YOLO corresponde a cada robot usando
                    # el último centroide conocido
                    last_donor  = self._find_last_centroid(outputs_per_frame, donor_id, fidx)
                    last_lost   = self._find_last_centroid(outputs_per_frame, lost_id, fidx)

                    # Asignar detecciones YOLO a donor y lost por distancia
                    def assign(dets, centroid):
                        if centroid is None or not dets:
                            return None
                        return min(dets, key=lambda d: _dist((d["cx"], d["cy"]), centroid))

                    donor_det = assign(robot_dets, last_donor)
                    lost_det  = assign(
                        [d for d in robot_dets if d is not donor_det], last_lost
                    )

                    if donor_det is None or lost_det is None:
                        continue

                    # Verificar que la posición del lost_det está dentro del donor_mask
                    lx, ly = lost_det["cx"], lost_det["cy"]
                    if ly >= frame_shape[0] or lx >= frame_shape[1]:
                        continue
                    if not donor_mask[ly, lx]:
                        continue  # no hay fusión real

                    # ── Corrección ──────────────────────────────────────────
                    # 1. Recortar donor_mask a su YOLO box
                    donor_box_mask = _box_mask(donor_det["box_xyxy"], frame_shape)
                    new_donor_mask = donor_mask & donor_box_mask

                    # 2. Máscara del lost = intersección del blob fusionado con su YOLO box
                    lost_box_mask  = _box_mask(lost_det["box_xyxy"], frame_shape)
                    new_lost_mask  = donor_mask & lost_box_mask

                    # Si la intersección es muy pequeña → usar la caja directamente
                    if new_lost_mask.sum() < 200:
                        new_lost_mask = lost_box_mask

                    # Aplicar corrección
                    frame_objs[donor_id]["mask"] = new_donor_mask
                    frame_objs[donor_id]["fusion_corrected"] = True

                    if lost_id not in frame_objs:
                        frame_objs[lost_id] = {
                            "label": obj_id_to_label.get(lost_id, f"obj_{lost_id}"),
                            "score": 0.0,
                        }
                    frame_objs[lost_id]["mask"]             = new_lost_mask
                    frame_objs[lost_id]["rescued"]          = True
                    frame_objs[lost_id]["yolo_box"]         = lost_det["box_xyxy"]
                    frame_objs[lost_id]["yolo_cx"]          = lost_det["cx"]
                    frame_objs[lost_id]["yolo_cy"]          = lost_det["cy"]
                    frame_objs[lost_id]["yolo_conf"]        = lost_det["conf"]
                    frame_objs[lost_id]["centroid_rescue"]  = (lost_det["cx"], lost_det["cy"])
                    frame_objs[lost_id]["fusion_corrected"] = True

                    fusion_events.append({
                        "frame_idx":  fidx,
                        "donor_id":   donor_id,
                        "lost_id":    lost_id,
                        "donor_px_before": donor_pixels,
                        "donor_px_after":  int(new_donor_mask.sum()),
                        "lost_px_after":   int(new_lost_mask.sum()),
                    })
                    fixed_count += 1
                    break  # solo corregir una fusión por frame/objeto

        if fusion_events:
            print(f"[YoloSam3Tracker] Fusión detectada y corregida: {fixed_count} frames")
        return outputs_per_frame, fusion_events


def apply_yolo_rescue_to_tracks_json(tracks_json: dict, rescue_events: list, metrics: dict) -> dict:
    """
    Añade rescue_events y métricas al JSON de tracks ya generado.
    Actualiza centroid de frames rescatados con el centroide de YOLO.
    """
    # Enriquecer centroids en frames rescatados
    for ev in rescue_events:
        fidx_str = str(ev["frame_idx"])
        obj_str = str(ev["obj_id"])
        if fidx_str in tracks_json and obj_str in tracks_json[fidx_str]:
            cx = (ev["yolo_box"][0] + ev["yolo_box"][2]) // 2
            cy = (ev["yolo_box"][1] + ev["yolo_box"][3]) // 2
            tracks_json[fidx_str][obj_str]["centroid"] = [cx, cy]
            tracks_json[fidx_str][obj_str]["centroid_rescue"] = [cx, cy]
            tracks_json[fidx_str][obj_str]["rescued"] = True
            tracks_json[fidx_str][obj_str]["yolo_conf"] = ev["yolo_conf"]

    # Añadir sección de métricas al JSON
    tracks_json["_yolo_rescue_metrics"] = metrics
    tracks_json["_yolo_rescue_events"] = rescue_events

    return tracks_json
