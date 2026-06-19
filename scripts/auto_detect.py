"""
Detección automática de robots y pelota en frames de FutBotMX.

Estrategia de ensemble:
  1. Pelota  → umbral HSV naranja (muy confiable en campo verde)
  2. Robots  → sustracción de fondo verde + naranja; contornos restantes
  3. YOLO    → confirma/complementa con 'sports ball' y detecciones generales

Uso:
    python scripts/auto_detect.py --frame output/frames/IMG_9866/00000.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np

# Ruta al modelo YOLO custom entrenado con robots Zumo + pelota.
# Si no existe, auto_detect cae al modo HSV.
CUSTOM_YOLO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "runs", "detect", "train-2", "weights", "best.pt"
)

# Umbrales HSV — usados solo como fallback si YOLO no encuentra nada
BALL_HSV_LOWER  = np.array([3,  100, 100])
BALL_HSV_UPPER  = np.array([30, 255, 255])

FIELD_HSV_LOWER = np.array([50,  40, 50])
FIELD_HSV_UPPER = np.array([100, 255, 255])

BALL_MIN_AREA   = 100
BALL_MAX_AREA   = 5000
ROBOT_MIN_AREA  = 1000
ROBOT_MAX_AREA  = 25000
ROBOT_MAX_RATIO = 3.5
BORDER_MARGIN   = 40
ROBOT_BOX_PAD   = 0.10

# ---------------------------------------------------------------------------

def _touches_border(x, y, w, h, frame_h, frame_w, margin=None):
    margin = margin or BORDER_MARGIN
    return x <= margin or y <= margin or (x+w) >= frame_w-margin or (y+h) >= frame_h-margin


def detect_ball_hsv(frame_bgr: np.ndarray):
    """
    Detecta la pelota naranja por umbral HSV.
    Elige el blob más pequeño, circular y lejos del borde (no el más grande).
    Retorna {"label":"ball", "cx", "cy", "box_xyxy"} o None.
    """
    H, W = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < BALL_MIN_AREA or area > BALL_MAX_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if _touches_border(x, y, bw, bh, H, W):
            continue
        ratio = max(bw, bh) / max(min(bw, bh), 1)
        if ratio > 2.5:   # pelota es casi circular
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # Score: más circular (ratio→1) y área dentro del rango esperado
        circularity = 4 * np.pi * area / (cv2.arcLength(c, True) ** 2 + 1e-6)
        candidates.append((circularity, area, cx, cy, x, y, bw, bh))

    if not candidates:
        return None

    # Preferir el más circular; desempate: el más pequeño (bola, no robot)
    candidates.sort(key=lambda t: (-t[0], t[1]))
    _, area, cx, cy, x, y, bw, bh = candidates[0]
    return {"label": "ball", "cx": cx, "cy": cy, "box_xyxy": [x, y, x+bw, y+bh]}


def get_field_mask(frame_bgr: np.ndarray):
    """Retorna máscara binaria del campo verde (zona de juego)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, FIELD_HSV_LOWER, FIELD_HSV_UPPER)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    # Dilata para incluir robots que están en el borde del campo
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
    mask = cv2.dilate(mask, k_dilate)
    return mask


def detect_robots_bg_subtraction(frame_bgr: np.ndarray, ball_box=None):
    """
    Detecta robots dentro del campo verde quitando el fondo.
    Retorna lista de {"label":"robotN", "cx", "cy", "box_xyxy", "area"}.
    """
    H, W = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    field_mask  = cv2.inRange(hsv, FIELD_HSV_LOWER, FIELD_HSV_UPPER)
    orange_mask = cv2.inRange(hsv, BALL_HSV_LOWER,  BALL_HSV_UPPER)

    # Foreground = no-verde y no-naranja
    fg = cv2.bitwise_not(cv2.bitwise_or(field_mask, orange_mask))

    # Solo considerar dentro del área de juego (campo + margen)
    field_roi = get_field_mask(frame_bgr)
    fg = cv2.bitwise_and(fg, field_roi)

    # Limpieza morfológica — kernel grande para unir cabeza+cuerpo del robot
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k_open)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close)

    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    robots = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < ROBOT_MIN_AREA or area > ROBOT_MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if _touches_border(x, y, w, h, H, W):
            continue
        ratio = max(w, h) / max(min(w, h), 1)
        if ratio > ROBOT_MAX_RATIO:
            continue
        # Expandir bbox para compensar detección parcial (cuerpo filtrado como campo/naranja)
        pad_w = int(w * ROBOT_BOX_PAD)
        pad_h = int(h * ROBOT_BOX_PAD)
        x1p = max(0, x - pad_w)
        y1p = max(0, y - pad_h)
        x2p = min(W, x + w + pad_w)
        y2p = min(H, y + h + pad_h)
        cx = (x1p + x2p) // 2
        cy = (y1p + y2p) // 2
        robots.append({"cx": cx, "cy": cy, "box_xyxy": [x1p, y1p, x2p, y2p], "area": float(area)})

    # Mayor área primero → robot1, robot2, …
    robots.sort(key=lambda r: r["area"], reverse=True)
    for i, r in enumerate(robots):
        r["label"] = f"robot{i+1}"

    return robots


def detect_with_custom_yolo(frame_bgr: np.ndarray, conf: float = 0.25):
    """
    Detecta robots y pelota con el modelo YOLO entrenado en robots Zumo.
    Retorna (robots, ball) donde:
      robots = [{"label":"robot1",...}, {"label":"robot2",...}, ...]  ordenados por área desc
      ball   = {"label":"ball", "cx", "cy", "box_xyxy"} o None
    Retorna (None, None) si el modelo no está disponible.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return None, None

    model_path = os.path.abspath(CUSTOM_YOLO_PATH)
    if not os.path.exists(model_path):
        return None, None

    if not hasattr(detect_with_custom_yolo, "_model"):
        detect_with_custom_yolo._model = YOLO(model_path)

    results = detect_with_custom_yolo._model(frame_bgr, conf=conf, verbose=False)

    robots_raw = []
    ball = None

    for r in results:
        for box in r.boxes:
            cls_name = detect_with_custom_yolo._model.names[int(box.cls[0])]
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            det = {"cx": cx, "cy": cy, "box_xyxy": [x1, y1, x2, y2],
                   "conf": float(box.conf[0]), "area": area}
            if cls_name == "ball":
                if ball is None or det["conf"] > ball["conf"]:
                    ball = det
            elif cls_name == "robot":
                robots_raw.append(det)

    # Ordenar robots por área descendente → robot1 es el más grande
    robots_raw.sort(key=lambda d: d["area"], reverse=True)
    robots = []
    for i, r in enumerate(robots_raw):
        robots.append({
            "label": f"robot{i+1}",
            "cx": r["cx"], "cy": r["cy"],
            "box_xyxy": r["box_xyxy"],
        })

    if ball:
        ball = {"label": "ball", "cx": ball["cx"], "cy": ball["cy"],
                "box_xyxy": ball["box_xyxy"]}

    return robots, ball


# ---------------------------------------------------------------------------

def auto_detect(frame_bgr: np.ndarray, use_yolo: bool = True):
    """
    Detección automática completa.
    Prioridad: modelo YOLO custom (robots Zumo) → fallback HSV si YOLO no disponible.
    Retorna lista ordenada: robots primero (robot1, robot2, …), pelota al final.
    Cada elemento: {"label", "cx", "cy", "box_xyxy"}
    """
    if use_yolo:
        robots, ball = detect_with_custom_yolo(frame_bgr)
        if robots is not None:
            # YOLO custom disponible — úsalo como fuente principal
            results = robots[:]
            if ball:
                results.append(ball)
            elif not ball:
                # YOLO no encontró pelota → intentar con HSV como respaldo
                ball_hsv = detect_ball_hsv(frame_bgr)
                if ball_hsv:
                    results.append(ball_hsv)
            return results

    # Fallback completo: HSV background subtraction
    ball = detect_ball_hsv(frame_bgr)
    robots = detect_robots_bg_subtraction(frame_bgr)
    results = robots[:]
    if ball:
        results.append(ball)
    for det in results:
        det.pop("area", None)
    return results


def draw_detections(frame_bgr: np.ndarray, detections: list) -> np.ndarray:
    """Dibuja los bounding boxes y etiquetas sobre el frame."""
    vis = frame_bgr.copy()
    colors = {
        "ball":   (0, 200, 255),
        "robot1": (255, 80,  0),
        "robot2": (0,  80, 255),
        "robot3": (0, 200,  50),
        "robot4": (200, 0, 200),
    }
    for det in detections:
        label = det["label"]
        x1, y1, x2, y2 = det["box_xyxy"]
        color = colors.get(label, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, (det["cx"], det["cy"]), 5, color, -1)
        cv2.putText(vis, f"{label} ({det['cx']},{det['cy']})",
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame", required=True, help="Ruta al frame (jpg/png)")
    parser.add_argument("--no_yolo", action="store_true", help="No usar YOLO")
    parser.add_argument("--out", default="", help="Guardar imagen de detección (opcional)")
    args = parser.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        print(f"No se pudo leer: {args.frame}")
        sys.exit(1)

    detections = auto_detect(frame, use_yolo=not args.no_yolo)

    print(f"\nDetectados: {len(detections)} objetos")
    for d in detections:
        print(f"  {d['label']:10s}  centro=({d['cx']:4d},{d['cy']:4d})  "
              f"box={d['box_xyxy']}")

    print("\nComando sugerido para pipeline.py:")
    pts = " ".join(f"{d['label']}:{d['cx']},{d['cy']}" for d in detections)
    print(f"  python scripts/pipeline.py --frames_dir <dir> --point_prompts \"{pts}\" --verify")

    vis = draw_detections(frame, detections)
    if args.out:
        cv2.imwrite(args.out, vis)
        print(f"\nImagen guardada: {args.out}")
    else:
        out_path = args.frame.replace(".jpg", "_detect.jpg").replace(".png", "_detect.png")
        cv2.imwrite(out_path, vis)
        print(f"\nImagen guardada: {out_path}")


if __name__ == "__main__":
    main()
