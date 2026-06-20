#!/usr/bin/env python3
"""
Analytics engine para Copa FutBotMX.
Calcula estadísticas por frame y resumen desde el JSON de tracks.

Métricas:
  · Posesión   — robot más cercano al balón (< POSSESSION_DIST_PX)
  · Velocidad  — |Δcentroide| / Δt  (px/s)
  · Distancia  — odómetro acumulado por objeto
  · Eventos    — pase (cambio de posesión), colisión (IoU de cajas > umbral), gol

Goles:
  Requiere --corners (JSON de esquinas del campo) para proyectar el balón al
  espacio cenital.  Las porterías se definen como los extremos izquierdo/derecho
  del campo cenital.  --team_left indica qué robot defiende la portería izquierda
  (el que mete gol en la portería contraria es el anotador).

Uso:
    python scripts/analytics.py \
        --tracks     output/tracks/IMG_9866_tracks.json \
        --frames_dir output/frames/IMG_9866 \
        --fps 30 --step 3
    # Salida automática: output/analytics/IMG_9866_analytics.json
"""

import argparse
import json
import math
import os

import cv2
import numpy as np

# ── Umbrales ───────────────────────────────────────────────────────────────────

POSSESSION_DIST_PX   = 150    # px entre centroide robot-balón → posesión
COLLISION_IOU_THRESH = 0.05   # IoU mínimo entre cajas de robots → colisión
INTERCEPTION_DIST_PX = 200    # px entre robots para clasificar cambio de posesión como intercepción
MIN_SCORE            = 0.3    # score SAM3/YOLO mínimo para usar centroide
EVENT_MIN_GAP        = 3      # frames mínimos entre eventos del mismo tipo
GOAL_COOLDOWN        = 30     # frames mínimos entre dos goles del mismo lado
GOAL_BBOX_MARGIN     = 80     # px de margen alrededor del bbox de portería
SHOT_HISTORY         = 6      # frames de historial para ajustar trayectoria
SHOT_LOOKAHEAD       = 18     # frames a extrapolar para predicción de tiro
SHOT_MARGIN          = 100    # px de margen alrededor del bbox al proyectar
SHOT_MIN_SPEED       = 8.0    # px/frame mínima para considerar tiro (descarta movimiento lento)
SHOT_COOLDOWN        = 25     # frames mínimos entre eventos shot_on_goal

# Dimensiones del canvas cenital (deben coincidir con visualize.py)
FIELD_W, FIELD_H = 800, 540


# ── Geometría ─────────────────────────────────────────────────────────────────

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _iou(box1, box2):
    if not box1 or not box2:
        return 0.0
    x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter)


def build_color_goal_detector(frames_dir, frame_paths, tracks, robot_labels):
    """
    Detecta las porterías por color (amarilla / azul) usando mediana temporal:
    los píxeles de portería son estables entre frames; la ropa de los jugadores
    cambia de posición y desaparece en la mediana.

    Retorna (check_goal_fn, goal_info_dict) donde:
      check_goal_fn(ball_pos, frame_bgr) → ("color", "side", scorer) o (None,None,None)
      goal_info_dict: metadatos de las porterías detectadas (para debug/visualización)
    """
    N_FRAMES    = min(20, len(frame_paths))
    MIN_AREA    = 3000   # px² — ignorar blobs pequeños (ruido)
    BALL_RADIUS = 40     # px — región alrededor del centroide del balón a muestrear

    # HSV ranges
    HSV_YELLOW = ((15, 100, 100), (38, 255, 255))
    HSV_BLUE   = ((95, 100,  40), (130, 255, 255))

    print(f"  Detectando porterías por color sobre {N_FRAMES} frames…")
    stacks = {"yellow": [], "blue": []}
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

    for fp in frame_paths[:N_FRAMES]:
        frame = cv2.imread(fp)
        if frame is None:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for color, (lo, hi) in [("yellow", HSV_YELLOW), ("blue", HSV_BLUE)]:
            m = cv2.inRange(hsv, np.array(lo), np.array(hi))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
            stacks[color].append(m)

    if not stacks["yellow"]:
        print("  ADVERTENCIA: No se pudieron cargar frames para detección de porterías")
        return None, {}

    goal_info = {}
    goal_masks = {}

    H_frame, W_frame = cv2.imread(frame_paths[0]).shape[:2]

    for color, frames_list in stacks.items():
        # Mediana temporal: solo píxeles estables (portería real, no jugadores)
        med = np.median(np.array(frames_list), axis=0).astype(np.uint8)
        _, stable = cv2.threshold(med, 127, 255, cv2.THRESH_BINARY)

        # Blob más grande = portería principal
        cnts, _ = cv2.findContours(stable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= MIN_AREA]
        if not cnts:
            print(f"  ADVERTENCIA: no se detectó portería {color}")
            continue

        best = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(best)
        M_  = cv2.moments(best)
        cx  = int(M_["m10"] / M_["m00"]) if M_["m00"] > 0 else x + w // 2
        side = "left" if cx < W_frame // 2 else "right"

        # Máscara de la portería (solo el blob principal)
        mask = np.zeros((H_frame, W_frame), dtype=np.uint8)
        cv2.drawContours(mask, [best], -1, 255, -1)
        # Dilatar levemente para tolerar jitter del balón
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

        goal_masks[color] = mask
        goal_info[color]  = {
            "side": side, "cx": cx, "bbox": [x, y, x+w, y+h], "area": int(cv2.contourArea(best))
        }
        print(f"  Portería {color}: lado={side}  cx={cx}  bbox=[{x},{y},{x+w},{y+h}]")

    if not goal_masks:
        return None, {}

    # ── Auto-asignar defensor por posición inicial ──────────────────────────
    # El robot que arranca más cerca de cada portería la defiende
    first_fidx = sorted(tracks.keys(), key=int)[0]
    first_frame = tracks[first_fidx]
    scorer_for  = {}   # color → robot que ANOTA ahí (el que no la defiende)

    for color, info in goal_info.items():
        gcx = info["cx"]
        best_lbl, best_dist = None, float("inf")
        for lbl in robot_labels:
            obj = first_frame.get(lbl, {})
            if obj.get("centroid"):
                d = abs(obj["centroid"][0] - gcx)
                if d < best_dist:
                    best_dist, best_lbl = d, lbl
        # best_lbl defiende este arco → anota el otro
        other = next((r for r in robot_labels if r != best_lbl), robot_labels[0])
        scorer_for[color] = other
        print(f"  Portería {color}: defiende={best_lbl}  anota={other}")

    # ── Función de detección por frame ──────────────────────────────────────
    def check_goal(ball_pos, frame_bgr):
        """
        Comprueba si el balón está dentro de una portería en este frame.
        Usa DOS criterios (ambos deben cumplirse):
          1. El centroide del balón está en la máscara estable de la portería
          2. El color dominante alrededor del balón es amarillo o azul
        Retorna (color, side, scorer) o (None, None, None).
        """
        cx, cy = int(ball_pos[0]), int(ball_pos[1])
        H_, W_ = frame_bgr.shape[:2]
        if not (0 <= cx < W_ and 0 <= cy < H_):
            return None, None, None

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        for color, mask in goal_masks.items():
            # Criterio 1: centroide dentro de la máscara estable O cerca del bbox
            in_mask = (0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]
                       and mask[cy, cx] != 0)
            if not in_mask:
                x1, y1, x2, y2 = goal_info[color]["bbox"]
                m = GOAL_BBOX_MARGIN
                if not (x1 - m <= cx <= x2 + m and y1 - m <= cy <= y2 + m):
                    continue

            # Criterio 2: color en la vecindad del balón
            r  = BALL_RADIUS
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(W_, cx + r), min(H_, cy + r)
            region = hsv[y1:y2, x1:x2]
            if region.size == 0:
                continue

            lo, hi = (HSV_YELLOW if color == "yellow" else HSV_BLUE)
            color_pct = (cv2.inRange(region, np.array(lo), np.array(hi)) > 0).mean()

            if color_pct > 0.20:   # ≥20% del área del balón en color de portería
                info   = goal_info[color]
                scorer = scorer_for.get(color, robot_labels[-1])
                return color, info["side"], scorer

        return None, None, None

    return check_goal, goal_info


# ── Predicción de tiro a gol por trayectoria ─────────────────────────────────

def _predict_trajectory(ball_history, goal_bboxes):
    """
    Ajusta una recta a los últimos SHOT_HISTORY centroides del balón y la
    extrapola SHOT_LOOKAHEAD frames hacia adelante.
    Retorna (color, frames_to_goal) si la proyección cae cerca de alguna
    portería, o (None, None) si no hay tiro.
    """
    pts = np.array(ball_history[-SHOT_HISTORY:], dtype=float)
    n   = len(pts)
    if n < 3:
        return None, None

    t  = np.arange(n, dtype=float)
    vx = np.polyfit(t, pts[:, 0], 1)[0]   # px por frame
    vy = np.polyfit(t, pts[:, 1], 1)[0]

    speed = math.hypot(vx, vy)
    if speed < SHOT_MIN_SPEED:
        return None, None

    last = pts[-1]
    for k in range(1, SHOT_LOOKAHEAD + 1):
        px = last[0] + vx * k
        py = last[1] + vy * k
        for color, bbox in goal_bboxes.items():
            x1, y1, x2, y2 = bbox
            m = SHOT_MARGIN
            if x1 - m <= px <= x2 + m and y1 - m <= py <= y2 + m:
                return color, k

    return None, None


# ── Análisis principal ────────────────────────────────────────────────────────

def analyze(tracks: dict, fps: int = 30, step: int = 3,
            frames_dir: str = "", frame_paths: list = None,
            corners_path: str = "", team_left: str = "robot1") -> dict:
    """
    Retorna:
      {
        "frames":  { "fidx": per_frame_dict },
        "summary": { posesión, velocidades, distancias, … },
        "events":  [ event_dict, … ],
        "paths":   { label: [(cx,cy), …] }
      }
    """
    dt     = step / fps                  # segundos entre frames analizados
    fidxs  = sorted(tracks.keys(), key=int)

    # Determinar labels de robots (todo lo que no es "ball")
    first  = tracks[fidxs[0]]
    robot_labels = sorted(k for k in first if k.startswith("robot"))
    all_labels   = robot_labels + ["ball"]

    # ── Detector de gol por color (visión por computadora) ────────────────────
    goal_scores    = {lbl: 0 for lbl in robot_labels}
    check_goal_fn  = None
    goal_info_meta = {}
    goal_bboxes    = {}   # color → (x1,y1,x2,y2) para predicción de trayectoria
    ball_history   = []   # historial de centroides del balón

    if frames_dir or frame_paths:
        import glob as _glob, os as _os
        if frame_paths is None:
            frame_paths = sorted(_glob.glob(_os.path.join(frames_dir, "*.jpg")) +
                                 _glob.glob(_os.path.join(frames_dir, "*.png")))
        if frame_paths:
            check_goal_fn, goal_info_meta = build_color_goal_detector(
                frames_dir, frame_paths, tracks, robot_labels)
            goal_bboxes = {c: tuple(info["bbox"]) for c, info in goal_info_meta.items()}

    # Precargar paths de frames por índice para acceso rápido dentro del loop
    _fidx_to_path = {}
    if frame_paths:
        for fp in frame_paths:
            fname = os.path.basename(fp)
            try:
                idx = int(os.path.splitext(fname)[0])
                _fidx_to_path[idx] = fp
            except ValueError:
                pass

    ball_in_goal_prev = False   # estado anterior (para detectar cruce de línea)

    frames_out = {}
    events     = []
    paths      = {lbl: [] for lbl in all_labels}

    prev_possessor = None
    prev_state     = None          # {label: pos}
    event_last     = {}            # event_key → last frame

    for fidx_str in fidxs:
        fidx  = int(fidx_str)
        frame = tracks[fidx_str]

        # ── Leer posiciones del frame ──────────────────────────────────────
        ball_obj = frame.get("ball", {})
        ball_pos = (
            tuple(ball_obj["centroid"])
            if ball_obj.get("centroid") and ball_obj.get("score", 0) >= MIN_SCORE
            else None
        )

        robot_pos = {}
        robot_box = {}
        for lbl in robot_labels:
            obj = frame.get(lbl, {})
            if obj.get("centroid"):
                robot_pos[lbl] = tuple(obj["centroid"])
                robot_box[lbl] = obj.get("box_xyxy")

        # Registrar trayectorias
        if ball_pos:
            paths["ball"].append(list(ball_pos))
        for lbl, pos in robot_pos.items():
            paths[lbl].append(list(pos))

        # ── Velocidades (px/s) ─────────────────────────────────────────────
        vels = {}
        if prev_state:
            for lbl in robot_labels:
                p0 = prev_state.get(lbl)
                p1 = robot_pos.get(lbl)
                vels[lbl] = round(_dist(p0, p1) / dt, 1) if p0 and p1 else 0.0
            p0b = prev_state.get("ball")
            vels["ball"] = round(_dist(p0b, ball_pos) / dt, 1) if p0b and ball_pos else 0.0
        else:
            vels = {lbl: 0.0 for lbl in all_labels}

        # ── Posesión ───────────────────────────────────────────────────────
        ball_dists = {}
        possessor  = None
        if ball_pos:
            for lbl in robot_labels:
                if lbl in robot_pos:
                    ball_dists[lbl] = round(_dist(ball_pos, robot_pos[lbl]), 1)
            if ball_dists:
                closest = min(ball_dists, key=ball_dists.get)
                if ball_dists[closest] < POSSESSION_DIST_PX:
                    possessor = closest

        # ── Detección de eventos ───────────────────────────────────────────
        frame_events = []

        # Pase / Intercepción: cambio de poseedor entre dos robots (no desde/hacia "none")
        # — Intercepción: robots cerca entre sí o cajas solapadas → el rival "roba" el balón
        # — Pase: robots separados → transferencia deliberada
        if possessor != prev_possessor:
            if possessor is not None and prev_possessor is not None:
                pos_a = robot_pos.get(prev_possessor)
                pos_b = robot_pos.get(possessor)
                inter_dist = _dist(pos_a, pos_b) if pos_a and pos_b else float("inf")
                iou_val    = _iou(robot_box.get(prev_possessor), robot_box.get(possessor))
                ev_type    = ("interception"
                              if inter_dist < INTERCEPTION_DIST_PX or iou_val > COLLISION_IOU_THRESH
                              else "pass")
                last = event_last.get(ev_type, -999)
                if fidx - last >= EVENT_MIN_GAP:
                    ev = {
                        "type":       ev_type, "frame": fidx,
                        "time_s":     round(fidx * dt, 2),
                        "from":       prev_possessor, "to": possessor,
                        "robot_dist": round(inter_dist, 1),
                    }
                    events.append(ev)
                    frame_events.append(ev)
                    event_last[ev_type] = fidx
            prev_possessor = possessor

        # Colisión: IoU entre cajas de robots
        for i, la in enumerate(robot_labels):
            for lb in robot_labels[i + 1:]:
                box_iou = _iou(robot_box.get(la), robot_box.get(lb))
                if box_iou > COLLISION_IOU_THRESH:
                    key  = f"coll_{la}_{lb}"
                    last = event_last.get(key, -999)
                    if fidx - last >= EVENT_MIN_GAP:
                        ev = {
                            "type": "collision", "frame": fidx,
                            "time_s": round(fidx * dt, 2),
                            "robots": [la, lb], "iou": round(box_iou, 3),
                        }
                        events.append(ev)
                        frame_events.append(ev)
                        event_last[key] = fidx

        # ── Gol por visión de color ────────────────────────────────────────
        if check_goal_fn and ball_pos:
            frame_bgr = cv2.imread(_fidx_to_path[fidx]) if fidx in _fidx_to_path else None
            if frame_bgr is not None:
                goal_color, goal_side, scorer = check_goal_fn(ball_pos, frame_bgr)
                ball_in_goal_now = goal_color is not None

                # Solo disparar en la transición campo→portería (cruce de línea)
                if ball_in_goal_now and not ball_in_goal_prev:
                    key  = f"goal_{goal_side}"
                    last = event_last.get(key, -999)
                    if fidx - last >= GOAL_COOLDOWN:
                        goal_scores[scorer] = goal_scores.get(scorer, 0) + 1
                        ev = {
                            "type":     "goal", "frame": fidx,
                            "time_s":   round(fidx * dt, 2),
                            "color":    goal_color,   # "yellow" o "blue"
                            "side":     goal_side,    # "left" o "right"
                            "scorer":   scorer,
                            "score":    dict(goal_scores),
                        }
                        events.append(ev)
                        frame_events.append(ev)
                        event_last[key] = fidx
                        print(f"  ⚽ GOL frame {fidx:04d} ({ev['time_s']}s) — "
                              f"{scorer} (portería {goal_color}/{goal_side})  "
                              f"marcador: {goal_scores}")

                ball_in_goal_prev = ball_in_goal_now

        # ── Actualizar historial y predecir tiro a gol ─────────────────────
        if ball_pos:
            ball_history.append(list(ball_pos))
            if len(ball_history) > SHOT_HISTORY:
                ball_history = ball_history[-SHOT_HISTORY:]

        if goal_bboxes and len(ball_history) >= 3:
            shot_color, frames_to_goal = _predict_trajectory(ball_history, goal_bboxes)
            if shot_color is not None:
                key  = f"shot_{shot_color}"
                last = event_last.get(key, -999)
                if fidx - last >= SHOT_COOLDOWN:
                    shot_info = goal_info_meta.get(shot_color, {})
                    ev = {
                        "type":           "shot_on_goal",
                        "frame":          fidx,
                        "time_s":         round(fidx * dt, 2),
                        "color":          shot_color,
                        "side":           shot_info.get("side", "?"),
                        "frames_to_goal": frames_to_goal,
                    }
                    events.append(ev)
                    frame_events.append(ev)
                    event_last[key] = fidx
                    print(f"  🎯 TIRO  frame {fidx:04d} ({ev['time_s']}s) — "
                          f"portería {shot_color}/{ev['side']}  "
                          f"(impacto en ~{frames_to_goal} frames)")

        # ── Guardar frame ──────────────────────────────────────────────────
        frames_out[fidx] = {
            "possessor":  possessor,
            "ball_pos":   list(ball_pos) if ball_pos else None,
            "ball_dists": ball_dists,
            "velocities": vels,
            "events":     frame_events,
        }

        # Estado para el siguiente frame
        prev_state = {lbl: robot_pos.get(lbl) for lbl in robot_labels}
        prev_state["ball"] = ball_pos

    # ── Resumen ────────────────────────────────────────────────────────────
    total = len(fidxs)
    poss_counts = {lbl: 0 for lbl in robot_labels + ["none"]}
    speed_lists = {lbl: [] for lbl in all_labels}
    dist_totals = {lbl: 0.0  for lbl in all_labels}

    for fidx_str in fidxs:
        fidx = int(fidx_str)
        fe   = frames_out[fidx]
        poss_counts[fe["possessor"] or "none"] += 1
        for lbl, spd in fe["velocities"].items():
            speed_lists[lbl].append(spd)
            dist_totals[lbl] += spd * dt

    def _avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0.0

    by_type = {}
    for ev in events:
        by_type[ev["type"]] = by_type.get(ev["type"], 0) + 1

    summary = {
        "total_frames":      total,
        "total_duration_s":  round(total * dt, 1),
        "fps_analyzed":      round(fps / step, 2),
        "possession": {
            lbl: {
                "frames": poss_counts.get(lbl, 0),
                "pct":    round(100 * poss_counts.get(lbl, 0) / total, 1),
            }
            for lbl in robot_labels + ["none"]
        },
        "speed_avg_px_s": {lbl: _avg(speed_lists[lbl]) for lbl in all_labels},
        "distance_px":    {lbl: round(dist_totals[lbl], 1) for lbl in all_labels},
        "score":          dict(goal_scores),
        "total_events":   len(events),
        "events_by_type": by_type,
    }

    return {
        "frames":  {str(k): v for k, v in frames_out.items()},
        "summary": summary,
        "events":  events,
        "paths":   paths,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tracks",     required=True, help="JSON de tracks (salida de pipeline.py)")
    parser.add_argument("--output",     default="",
                        help="Ruta JSON de salida (default: output/analytics/<video>_analytics.json)")
    parser.add_argument("--fps",        type=int, default=30, help="FPS del video original")
    parser.add_argument("--step",       type=int, default=3,  help="Paso de extracción de frames")
    parser.add_argument("--frames_dir", default="",
                        help="Directorio de frames (activa detección de gol por color)")
    args = parser.parse_args()

    with open(args.tracks) as f:
        tracks = json.load(f)

    print(f"Analizando {len(tracks)} frames  (fps={args.fps}, step={args.step}) …")
    if args.frames_dir:
        print(f"  Detección de gol por COLOR activada  (frames_dir={args.frames_dir})")
    result = analyze(tracks, fps=args.fps, step=args.step,
                     frames_dir=args.frames_dir)

    # Ruta de salida: output/analytics/<video>_analytics.json por defecto
    if args.output:
        out_path = args.output
    else:
        video_name = os.path.basename(args.tracks).replace("_tracks.json", "")
        out_path = os.path.join("output", "analytics", f"{video_name}_analytics.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    s = result["summary"]
    print(f"\n── Resumen ─────────────────────────────────────────────────")
    print(f"  Duración       : {s['total_duration_s']} s  ({s['total_frames']} frames)")
    print(f"  Posesión:")
    for lbl, d in s["possession"].items():
        bar = "█" * int(d["pct"] / 5)
        print(f"    {lbl:10s}: {d['pct']:5.1f}%  {bar}")
    print(f"  Velocidad prom (px/s):")
    for lbl, v in s["speed_avg_px_s"].items():
        print(f"    {lbl:10s}: {v}")
    print(f"  Distancia total (px):")
    for lbl, d in s["distance_px"].items():
        print(f"    {lbl:10s}: {d}")
    if s.get("score"):
        marcador = "  ".join(f"{lbl} {g}" for lbl, g in s["score"].items())
        print(f"  Marcador       : {marcador}")
    print(f"  Eventos: {s['total_events']}  →  {s['events_by_type']}")
    for ev in result["events"]:
        if ev["type"] == "pass":
            print(f"    Pase      {ev['time_s']:6.2f}s  {ev['from']} → {ev['to']}")
        elif ev["type"] == "interception":
            print(f"    Intercep  {ev['time_s']:6.2f}s  {ev['from']} → {ev['to']}  "
                  f"(dist={ev['robot_dist']:.0f}px)")
        elif ev["type"] == "collision":
            print(f"    Colisión  {ev['time_s']:6.2f}s  {ev['robots']}  IoU={ev['iou']}")
        elif ev["type"] == "goal":
            print(f"    ⚽ GOL    {ev['time_s']:6.2f}s  {ev['scorer']}  (portería {ev['side']})  "
                  f"→ {ev['score']}")
        elif ev["type"] == "shot_on_goal":
            print(f"    🎯 TIRO   {ev['time_s']:6.2f}s  portería {ev['color']}/{ev['side']}  "
                  f"(impacto ~{ev['frames_to_goal']} frames)")
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
