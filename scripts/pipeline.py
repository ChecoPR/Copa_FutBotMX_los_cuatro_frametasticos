"""
Pipeline principal: segmentación y tracking de fútbol robótico.

Arquitectura (modo --auto):
  · Robots  → YOLO cada frame + tracking por proximidad (IDs consistentes)
  · Pelota  → SAM3 VG (caja o texto) → máscara precisa propagada
  · Fusión  → centroides YOLO (posición) + máscaras SAM3 (forma exacta)

Esta separación es la correcta: dos robots idénticos no se pueden distinguir
por apariencia visual; YOLO con tracking por proximidad sí mantiene la identidad.
SAM3 aporta la máscara de alta calidad para la pelota.

Comandos:
    # Modo automático (YOLO + SAM3 pelota)
    python scripts/pipeline.py --frames_dir output/frames/IMG_9866 --auto

    # Forzar también SAM3 para robot1 además de la pelota
    python scripts/pipeline.py --frames_dir output/frames/IMG_9866 --auto --sam3_robots

    # Modo manual (todas las sesiones SAM3)
    python scripts/pipeline.py --frames_dir output/frames/IMG_9866 \\
        --point_prompts "robot1:848,295 ball:1085,267 robot2:1204,162"
"""

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

SAM3_REPO = os.path.join(os.path.dirname(__file__), "..", "sam3")
BPE_PATH  = os.path.join(SAM3_REPO, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

LABEL_COLORS = {
    "robot1": (255, 100,   0),
    "robot2": (  0, 100, 255),
    "robot3": (  0, 200,  50),
    "ball":   (  0, 230, 230),
}
DEFAULT_COLOR = (180, 180, 180)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def build_predictor():
    from sam3.model_builder import build_sam3_video_predictor
    gpus = [0]
    print(f"Cargando SAM3 en GPU(s): {gpus}")
    predictor = build_sam3_video_predictor(bpe_path=BPE_PATH, gpus_to_use=gpus)
    print("Predictor listo.")
    return predictor


def get_frame_paths(frames_dir: str) -> list:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if os.path.splitext(f)[1].lower() in exts
    )


def parse_point_prompts(raw: str) -> list:
    entries = []
    for token in raw.strip().split():
        if ":" not in token:
            raise ValueError(f"Formato inválido: '{token}'. Usa label:x,y")
        label, coords = token.split(":", 1)
        x, y = coords.split(",")
        entries.append((label.strip(), float(x), float(y)))
    return entries


def mask_to_centroid(mask):
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = np.asarray(mask).astype(bool)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (int(xs.mean()), int(ys.mean()))


def mask_to_np(mask) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        return mask.cpu().numpy().astype(bool)
    return np.asarray(mask).astype(bool)


# ---------------------------------------------------------------------------
# Tracking YOLO por proximidad (robots)
# ---------------------------------------------------------------------------

def track_robots_yolo(frame_paths: list, yolo_model) -> dict:
    """
    Tracking con "zona de fusión" + extrapolación de trayectoria pre-encuentro.

    Estrategia anti-swap:
    1. Se estima velocidad con regresión lineal sobre los últimos HIST_LEN frames.
    2. Cuando dos robots están a < MERGE_DIST píxeles (zona de fusión), se congela
       el historial de velocidad y se extrapola linealmente desde la posición/velocidad
       ANTES de entrar a la zona.  Esto permite predecir correctamente qué robot
       saldrá a cada lado aunque ambos estén momentáneamente en el mismo punto.
    3. Asignación óptima global con algoritmo húngaro (scipy) en cada frame.

    Retorna {fidx: {"robot1": {cx, cy, box_xyxy, conf}, ...}}
    """
    try:
        from scipy.optimize import linear_sum_assignment
        _has_scipy = True
    except ImportError:
        _has_scipy = False
    from collections import deque

    HIST_LEN   = 12    # frames de historial para regresión de velocidad
    MAX_DIST   = 300   # distancia máxima predicción→detección para aceptar match
    MERGE_DIST = 160   # píxeles entre centroides para declarar "zona de fusión"

    # {label: {"pos":(cx,cy), "vel":(vx,vy), "hist": deque}}
    states = {}
    result = {}

    # Estado de zona de fusión (global por video)
    merge_count   = 0    # frames consecutivos en zona de fusión
    pre_merge_snap = {}  # snapshot de pos+vel al entrar a la zona

    def _vel_from_hist(hist):
        """Velocidad por regresión lineal sobre el historial de posiciones."""
        n = len(hist)
        if n < 3:
            return None
        t = list(range(n))
        xs = [p[0] for p in hist]
        ys = [p[1] for p in hist]
        t_m = sum(t) / n
        x_m = sum(xs) / n
        y_m = sum(ys) / n
        denom = sum((ti - t_m)**2 for ti in t) or 1e-9
        vx = sum((t[i]-t_m)*(xs[i]-x_m) for i in range(n)) / denom
        vy = sum((t[i]-t_m)*(ys[i]-y_m) for i in range(n)) / denom
        return (vx, vy)

    for fidx, fpath in enumerate(frame_paths):
        frame_bgr = cv2.imread(fpath)
        if frame_bgr is None:
            result[fidx] = {}
            continue

        yolo_results = yolo_model(frame_bgr, conf=0.25, verbose=False)
        robot_dets = []
        for r in yolo_results:
            for box in r.boxes:
                if yolo_model.names[int(box.cls[0])] != "robot":
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                robot_dets.append({
                    "cx": (x1+x2)//2, "cy": (y1+y2)//2,
                    "box_xyxy": [x1, y1, x2, y2],
                    "conf": float(box.conf[0]),
                    "area": (x2-x1)*(y2-y1),
                })
        robot_dets.sort(key=lambda d: d["area"], reverse=True)

        frame_robots = {}

        if not states:
            # Frame 0: asignar por área
            for i, det in enumerate(robot_dets[:3]):
                lbl  = f"robot{i+1}"
                pos  = (float(det["cx"]), float(det["cy"]))
                hist = deque([pos], maxlen=HIST_LEN)
                states[lbl] = {"pos": pos, "vel": (0.0, 0.0), "hist": hist}
                frame_robots[lbl] = det
        else:
            labels = sorted(states.keys())

            # Velocidades actuales estimadas por regresión
            cur_vels = {lbl: (_vel_from_hist(states[lbl]["hist"]) or states[lbl]["vel"])
                        for lbl in labels}

            # Predicción normal (pos + vel actual)
            naive_pred = {lbl: (states[lbl]["pos"][0] + cur_vels[lbl][0],
                                states[lbl]["pos"][1] + cur_vels[lbl][1])
                          for lbl in labels}

            # ¿Están los robots en zona de fusión?
            pred_pts = [naive_pred[lbl] for lbl in labels]
            in_merge = len(pred_pts) >= 2 and any(
                np.hypot(pred_pts[i][0]-pred_pts[j][0],
                         pred_pts[i][1]-pred_pts[j][1]) < MERGE_DIST
                for i in range(len(pred_pts))
                for j in range(i+1, len(pred_pts))
            )

            if in_merge:
                merge_count += 1
                if merge_count == 1:
                    # Primer frame en zona: guardar trayectoria previa
                    pre_merge_snap = {lbl: {"pos": states[lbl]["pos"],
                                            "vel": cur_vels[lbl]}
                                      for lbl in labels}
                    if len(labels) > 1:
                        print(f"  [merge] frame {fidx:04d}: robots cerca — "
                              f"extrapolando trayectorias pre-encuentro")
                # Predecir extrapolando desde el snapshot pre-fusión
                predicted = {
                    lbl: (pre_merge_snap[lbl]["pos"][0] + pre_merge_snap[lbl]["vel"][0] * merge_count,
                          pre_merge_snap[lbl]["pos"][1] + pre_merge_snap[lbl]["vel"][1] * merge_count)
                    for lbl in labels
                }
            else:
                if merge_count > 0:
                    print(f"  [merge] frame {fidx:04d}: robots separados tras {merge_count} frames")
                    merge_count    = 0
                    pre_merge_snap = {}
                predicted = naive_pred

            # Asignación óptima (húngara) o greedy de fallback
            n_lbl  = len(labels)
            n_dets = len(robot_dets)
            assignment = {}

            if n_dets > 0 and _has_scipy:
                n_cols = max(n_lbl, n_dets)
                cost   = np.full((n_lbl, n_cols), MAX_DIST * 2.0)
                for i, lbl in enumerate(labels):
                    px, py = predicted[lbl]
                    for j in range(min(n_dets, n_cols)):
                        cost[i, j] = np.hypot(robot_dets[j]["cx"]-px,
                                               robot_dets[j]["cy"]-py)
                rows, cols = linear_sum_assignment(cost)
                assignment = {labels[r]: cols[k]
                              for k, r in enumerate(rows)
                              if cols[k] < n_dets and cost[r, cols[k]] < MAX_DIST}
            elif n_dets > 0:
                available = list(range(n_dets))
                for lbl in labels:
                    if not available:
                        break
                    px, py = predicted[lbl]
                    best_j = min(available, key=lambda j: np.hypot(
                        robot_dets[j]["cx"]-px, robot_dets[j]["cy"]-py))
                    if np.hypot(robot_dets[best_j]["cx"]-px,
                                robot_dets[best_j]["cy"]-py) < MAX_DIST:
                        assignment[lbl] = best_j
                        available.remove(best_j)

            # Actualizar estados
            for lbl, j in assignment.items():
                det     = robot_dets[j]
                new_pos = (float(det["cx"]), float(det["cy"]))
                old_pos = states[lbl]["pos"]
                old_vel = states[lbl]["vel"]
                frame_robots[lbl] = det
                states[lbl]["pos"] = new_pos
                if not in_merge:
                    # Fuera de zona de fusión: actualizar historial y velocidad EMA
                    states[lbl]["hist"].append(new_pos)
                    new_vx = new_pos[0] - old_pos[0]
                    new_vy = new_pos[1] - old_pos[1]
                    states[lbl]["vel"] = (0.5*new_vx + 0.5*old_vel[0],
                                          0.5*new_vy + 0.5*old_vel[1])
                # En zona de fusión: actualizar pos pero NO hist/vel
                # (preserva la velocidad pre-encuentro para seguir extrapolando)

            for lbl in labels:
                if lbl not in assignment:
                    vx, vy = states[lbl]["vel"]
                    states[lbl]["vel"] = (vx*0.7, vy*0.7)

        result[fidx] = frame_robots

    return result


# ---------------------------------------------------------------------------
# Sesión SAM3 para un objeto
# ---------------------------------------------------------------------------

def run_sam3_session(predictor, frames_dir: str, box_norm: list, label: str,
                     init_frame: int = 0) -> dict:
    """
    Corre una sesión SAM3 completa para un objeto (VG + propagación).
    init_frame: índice del frame donde inicializar el prompt (default 0).
    Retorna {fidx: {"mask": np.ndarray|None, "score": float, "label": str}}
    """
    session_id = predictor.handle_request({
        "type": "start_session", "resource_path": frames_dir
    })["session_id"]

    predictor.handle_request({
        "type": "add_prompt", "session_id": session_id, "frame_index": init_frame,
        "bounding_boxes": [box_norm], "bounding_box_labels": [1],
    })

    session_out = {}
    for response in predictor.handle_stream_request({
        "type": "propagate_in_video", "session_id": session_id,
    }):
        fidx    = response["frame_index"]
        raw_out = response.get("outputs") or {}
        masks   = raw_out.get("out_binary_masks", [])
        probs   = raw_out.get("out_probs",   [])
        mask    = masks[0] if len(masks) > 0 else None
        score   = float(probs[0]) if len(probs) > 0 else 0.0
        session_out[fidx] = {"mask": mask, "score": score, "label": label}

    predictor.handle_request({"type": "close_session", "session_id": session_id})
    return session_out


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def render_overlay(frame_bgr: np.ndarray, frame_objs: dict, alpha: float = 0.45) -> np.ndarray:
    overlay = frame_bgr.copy()
    for label, obj_data in frame_objs.items():
        color = LABEL_COLORS.get(label, DEFAULT_COLOR)

        # Objeto YOLO: dibujar caja
        if obj_data.get("source") == "yolo":
            box = obj_data.get("box_xyxy")
            if box:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cx, cy = obj_data.get("cx", (x1+x2)//2), obj_data.get("cy", (y1+y2)//2)
                cv2.circle(overlay, (cx, cy), 5, color, -1)
                cv2.putText(overlay, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            continue

        # Objeto SAM3: dibujar máscara
        mask = mask_to_np(obj_data.get("mask"))
        if mask is None or not mask.any():
            # Fallback: dibujar caja YOLO si existe
            box = obj_data.get("box_xyxy")
            if box:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            continue

        overlay[mask] = (overlay[mask] * (1-alpha) + np.array(color) * alpha).astype(np.uint8)
        c = mask_to_centroid(mask)
        if c:
            cv2.circle(overlay, c, 5, color, -1)
            cv2.putText(overlay, label, (c[0]+7, c[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return overlay


def open_in_windows(path: str):
    try:
        r = subprocess.run(["wslpath", "-w", os.path.abspath(path)],
                           capture_output=True, text=True)
        subprocess.Popen(["explorer.exe", r.stdout.strip()])
        time.sleep(0.8)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(
    frames_dir: str,
    text_prompts: list,
    point_prompts: list,
    output_dir: str,
    fps: float = 10.0,
    verify: bool = False,
    auto: bool = False,
    no_yolo: bool = False,
    sam3_robots: bool = False,
    ball_point: tuple = None,
    ball_frame: int = 0,
):
    frame_paths = get_frame_paths(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"Sin frames en {frames_dir}")
    print(f"Frames: {len(frame_paths)}")

    frame0_bgr = cv2.imread(frame_paths[0])
    frame_H, frame_W = frame0_bgr.shape[:2]
    name = os.path.basename(frames_dir.rstrip("/"))

    # ── Detección automática en frame 0 ──────────────────────────────────
    detections_f0 = []
    if auto or point_prompts:
        if auto:
            from auto_detect import auto_detect, draw_detections
            print("Detección automática (YOLO)...")
            detections_f0 = auto_detect(frame0_bgr, use_yolo=not no_yolo)
            if not detections_f0:
                raise RuntimeError("Sin detecciones. Prueba --point_prompts manual.")
            print(f"  Objetos: {[d['label'] for d in detections_f0]}")

            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            det_path = os.path.join(debug_dir, f"{name}_autodetect.jpg")
            cv2.imwrite(det_path, draw_detections(frame0_bgr, detections_f0))
            print(f"  Imagen de detección: {det_path}")

            if verify:
                open_in_windows(det_path)
                for d in detections_f0:
                    print(f"    {d['label']:10s} cx={d['cx']:4d} cy={d['cy']:4d}")
                if input("  ¿Correcto? [s/N]: ").strip().lower() != "s":
                    return {}
        else:
            for label, cx, cy in point_prompts:
                detections_f0.append({"label": label, "cx": cx, "cy": cy, "box_xyxy": None})

    # Separar detecciones por tipo
    robot_dets_f0 = [d for d in detections_f0 if d["label"] != "ball"]
    ball_dets_f0  = [d for d in detections_f0 if d["label"] == "ball"]

    # --ball_point sobreescribe cualquier detección automática del balón
    if ball_point:
        cx, cy = ball_point
        hw = 40 / frame_W
        hh = 40 / frame_H
        ball_dets_f0 = [{"label": "ball", "cx": int(cx), "cy": int(cy),
                         "box_xyxy": [int(cx-40), int(cy-40), int(cx+40), int(cy+40)]}]
        print(f"[ball_point] Balón forzado en frame {ball_frame}: ({cx}, {cy})")

    # ── Salida unificada: {fidx: {label: obj_data}} ───────────────────────
    outputs_per_frame = {fidx: {} for fidx in range(len(frame_paths))}

    # ── SAM3 para la PELOTA ───────────────────────────────────────────────
    predictor = build_predictor()

    ball_det = ball_dets_f0[0] if ball_dets_f0 else None
    if ball_det:
        if ball_det.get("box_xyxy"):
            x1, y1, x2, y2 = ball_det["box_xyxy"]
            box_norm = [x1/frame_W, y1/frame_H, (x2-x1)/frame_W, (y2-y1)/frame_H]
        else:
            cx, cy = ball_det["cx"], ball_det["cy"]
            hw = 40 / frame_W
            hh = 40 / frame_H
            box_norm = [max(0., cx/frame_W - hw), max(0., cy/frame_H - hh),
                        min(2*hw, 1.0), min(2*hh, 1.0)]

        print(f"\n[SAM3] Pelota  frame={ball_frame}  box={[round(v,3) for v in box_norm]}")
        ball_session = run_sam3_session(predictor, frames_dir, box_norm, "ball",
                                        init_frame=ball_frame)
        for fidx, data in ball_session.items():
            outputs_per_frame[fidx]["ball"] = data
        print(f"  → {len(ball_session)} frames propagados")

    elif text_prompts:
        # Fallback: prompt de texto para la pelota
        session_id = predictor.handle_request({
            "type": "start_session", "resource_path": frames_dir
        })["session_id"]
        for prompt in text_prompts:
            print(f"  [TXT] '{prompt}'")
            predictor.handle_request({
                "type": "add_prompt", "session_id": session_id,
                "frame_index": 0, "text": prompt,
            })
        for response in predictor.handle_stream_request({
            "type": "propagate_in_video", "session_id": session_id,
        }):
            fidx    = response["frame_index"]
            raw_out = response.get("outputs") or {}
            masks   = raw_out.get("out_binary_masks", [])
            probs   = raw_out.get("out_probs",   [])
            mask    = masks[0] if len(masks) > 0 else None
            score   = float(probs[0]) if len(probs) > 0 else 0.0
            outputs_per_frame[fidx]["ball"] = {"mask": mask, "score": score, "label": "ball"}
        predictor.handle_request({"type": "close_session", "session_id": session_id})

    # ── SAM3 para ROBOT1 (opcional) ───────────────────────────────────────
    if sam3_robots and robot_dets_f0:
        r1 = robot_dets_f0[0]
        if r1.get("box_xyxy"):
            x1, y1, x2, y2 = r1["box_xyxy"]
            box_norm = [x1/frame_W, y1/frame_H, (x2-x1)/frame_W, (y2-y1)/frame_H]
        else:
            cx, cy = r1["cx"], r1["cy"]
            hw, hh = 60/frame_W, 80/frame_H
            box_norm = [max(0., cx/frame_W-hw), max(0., cy/frame_H-hh),
                        min(2*hw, 1.0), min(2*hh, 1.0)]

        print(f"\n[SAM3] Robot1  box={[round(v,3) for v in box_norm]}")
        r1_session = run_sam3_session(predictor, frames_dir, box_norm, "robot1")
        # Solo guardamos donde el score es bueno (>0.4)
        for fidx, data in r1_session.items():
            if data["score"] > 0.4:
                outputs_per_frame[fidx]["robot1_sam3"] = data

    # ── YOLO para ROBOTS en todos los frames ─────────────────────────────
    if robot_dets_f0 or auto:
        try:
            from ultralytics import YOLO as UltralyticsYOLO
            yolo_path = os.path.join(
                os.path.dirname(__file__), "..", "runs", "detect", "train-2", "weights", "best.pt"
            )
            yolo_path = os.path.abspath(yolo_path)
            if os.path.exists(yolo_path):
                print(f"\n[YOLO] Tracking robots en {len(frame_paths)} frames...")
                yolo_model = UltralyticsYOLO(yolo_path)
                robot_tracks = track_robots_yolo(frame_paths, yolo_model)

                for fidx, frame_robots in robot_tracks.items():
                    for label, det in frame_robots.items():
                        # Si ya tenemos máscara SAM3 para robot1, la fusionamos
                        if label == "robot1" and "robot1_sam3" in outputs_per_frame.get(fidx, {}):
                            sam3_data = outputs_per_frame[fidx].pop("robot1_sam3")
                            outputs_per_frame[fidx][label] = {
                                "mask":    sam3_data["mask"],
                                "score":   sam3_data["score"],
                                "label":   label,
                                "source":  "sam3",
                                "cx":      det["cx"],
                                "cy":      det["cy"],
                                "box_xyxy": det["box_xyxy"],
                            }
                        else:
                            outputs_per_frame[fidx][label] = {
                                "mask":    None,
                                "score":   det["conf"],
                                "label":   label,
                                "source":  "yolo",
                                "cx":      det["cx"],
                                "cy":      det["cy"],
                                "box_xyxy": det["box_xyxy"],
                            }
                n_robot1 = sum(1 for f in robot_tracks.values() if "robot1" in f)
                n_robot2 = sum(1 for f in robot_tracks.values() if "robot2" in f)
                print(f"  robot1: {n_robot1} frames  robot2: {n_robot2} frames")
            else:
                print(f"[YOLO] Modelo no encontrado: {yolo_path}")
        except ImportError:
            print("[YOLO] ultralytics no instalado")

    # ── Guardar máscaras ──────────────────────────────────────────────────
    masks_dir = os.path.join(output_dir, "masks", name)
    os.makedirs(masks_dir, exist_ok=True)
    label_to_id = {"robot1": 0, "robot2": 1, "robot3": 2, "ball": 3}
    for fidx, frame_objs in outputs_per_frame.items():
        for label, obj_data in frame_objs.items():
            m = mask_to_np(obj_data.get("mask"))
            if m is None:
                continue
            oid = label_to_id.get(label, 9)
            cv2.imwrite(
                os.path.join(masks_dir, f"{fidx:05d}_obj{oid}.png"),
                m.astype(np.uint8) * 255,
            )
    print(f"\nMáscaras: {masks_dir}/")

    # ── Guardar tracks JSON ───────────────────────────────────────────────
    tracks_dir = os.path.join(output_dir, "tracks")
    os.makedirs(tracks_dir, exist_ok=True)

    tracks_json = {}
    for fidx, frame_objs in outputs_per_frame.items():
        tracks_json[str(fidx)] = {}
        for label, obj_data in frame_objs.items():
            # Centroide: preferir explícito (YOLO) sobre el de la máscara
            cx = obj_data.get("cx")
            cy = obj_data.get("cy")
            if cx is None or cy is None:
                c = mask_to_centroid(obj_data.get("mask"))
                if c:
                    cx, cy = c
            tracks_json[str(fidx)][label] = {
                "label":    label,
                "score":    obj_data.get("score", 0.0),
                "source":   obj_data.get("source", "sam3"),
                "centroid": [cx, cy] if cx is not None else None,
                "box_xyxy": obj_data.get("box_xyxy"),
            }

    tracks_path = os.path.join(tracks_dir, f"{name}_tracks.json")
    with open(tracks_path, "w") as f:
        json.dump(tracks_json, f, indent=2)
    print(f"Tracks: {tracks_path}")

    # ── Renderizar video ──────────────────────────────────────────────────
    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    video_path = os.path.join(videos_dir, f"{name}_tracked.mp4")
    h, w = frame0_bgr.shape[:2]
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fidx, fpath in enumerate(frame_paths):
        frame = cv2.imread(fpath)
        if fidx in outputs_per_frame:
            frame = render_overlay(frame, outputs_per_frame[fidx])
        writer.write(frame)
    writer.release()
    print(f"Video: {video_path}")

    return tracks_json


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--prompts",       default="", help="Prompts de texto (coma)")
    parser.add_argument("--point_prompts", default="", help='"label:x,y ..."')
    parser.add_argument("--output_dir",
                        default=os.path.join(os.path.dirname(__file__), "..", "output"))
    parser.add_argument("--fps",         type=float, default=10.0)
    parser.add_argument("--verify",      action="store_true")
    parser.add_argument("--auto",        action="store_true",
                        help="Detección automática con YOLO")
    parser.add_argument("--no_yolo",     action="store_true",
                        help="Con --auto, usa solo HSV (sin YOLO)")
    parser.add_argument("--sam3_robots", action="store_true",
                        help="Agregar sesión SAM3 para robot1 (además de la pelota)")
    parser.add_argument("--ball_point", default="",
                        help="Coordenadas manuales del balón: x,y  (se puede combinar con --auto)")
    parser.add_argument("--ball_frame", type=int, default=0,
                        help="Frame de inicialización del balón para SAM3 (default 0)")
    args = parser.parse_args()

    text_prompts  = [p.strip() for p in args.prompts.split(",") if p.strip()]
    point_prompts = parse_point_prompts(args.point_prompts) if args.point_prompts.strip() else []

    if args.auto and (text_prompts or point_prompts):
        parser.error("--auto no se combina con --prompts / --point_prompts")
    if not args.auto and not text_prompts and not point_prompts and not args.ball_point:
        parser.error("Especifica --auto, --prompts, o --point_prompts")

    ball_point = None
    if args.ball_point:
        try:
            bx, by = args.ball_point.split(",")
            ball_point = (float(bx), float(by))
        except ValueError:
            parser.error("--ball_point debe ser x,y  (ej: 1085,267)")

    run_pipeline(
        args.frames_dir, text_prompts, point_prompts, args.output_dir, args.fps,
        verify=args.verify, auto=args.auto, no_yolo=args.no_yolo,
        sam3_robots=args.sam3_robots,
        ball_point=ball_point, ball_frame=args.ball_frame,
    )


if __name__ == "__main__":
    main()
