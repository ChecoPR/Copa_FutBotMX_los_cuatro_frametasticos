#!/usr/bin/env python3
"""
Detección automática de las 4 esquinas del campo.

Estrategia:
  1. Detecta porterías azul/amarilla por HSV → anclas con posición conocida.
  2. Detecta líneas blancas con HoughLines y las agrupa por ángulo.
  3. Intento A: prueba combinaciones de 4 líneas, valida con porterías.
  4. Intento B (fallback): usa porterías + línea superior como anclas
     para construir la homografía directamente, sin depender de corners
     que pueden quedar fuera del frame.
  5. Guarda JSON de corners + imagen de debug.

Uso:
    python scripts/auto_corners.py \
        --frame  output/frames/IMG_9866/00000.jpg \
        --out    output/field_corners_IMG_9866.json \
        --debug  output/debug_corners_IMG_9866.jpg
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# ── Parámetros ─────────────────────────────────────────────────────────────────
HSV_WHITE_LO  = np.array([0,   0, 155])
HSV_WHITE_HI  = np.array([180, 60, 255])
HSV_YELLOW_LO = np.array([15, 100, 100])
HSV_YELLOW_HI = np.array([38, 255, 255])
HSV_BLUE_LO   = np.array([95, 100,  40])
HSV_BLUE_HI   = np.array([130, 255, 255])

MORPH_K       = 8
FIELD_W       = 800
FIELD_H       = 540
EDGE_SNAP_PX  = 12   # si una esquina cae a < N px del borde del frame, se ancla al borde


# ── Utilidades ─────────────────────────────────────────────────────────────────

def _snap_corners(corners, W, H, tol=EDGE_SNAP_PX):
    """Ancla al borde del frame las esquinas que caen dentro de la tolerancia."""
    out = {}
    for name, pt in corners.items():
        x, y = float(pt[0]), float(pt[1])
        if x < tol:           x = 0.0
        elif x > W - tol:     x = float(W)
        if y < tol:           y = 0.0
        elif y > H - tol:     y = float(H)
        out[name] = [x, y]
    return out

def _intersect(r1, t1, r2, t2):
    A = np.array([[np.cos(t1), np.sin(t1)],
                  [np.cos(t2), np.sin(t2)]], dtype=float)
    b = np.array([r1, r2], dtype=float)
    try:
        x, y = np.linalg.solve(A, b)
        return (float(x), float(y)) if np.isfinite(x) and np.isfinite(y) else None
    except np.linalg.LinAlgError:
        return None


def _order_corners(pts):
    arr = np.array(pts, dtype=np.float32)
    s, d = arr[:, 0] + arr[:, 1], arr[:, 0] - arr[:, 1]
    return arr[np.argmin(s)], arr[np.argmax(d)], arr[np.argmax(s)], arr[np.argmin(d)]


def _detect_goal(frame, lo, hi, min_area=2000):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not cnts:
        return None
    best = max(cnts, key=cv2.contourArea)
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None
    return (M["m10"] / M["m00"], M["m01"] / M["m00"])


# ── Intento B: silueta del área verde ──────────────────────────────────────────

def _corners_from_green_hull(frame, goal_y, goal_b, verbose=True):
    """
    Detecta las 4 esquinas del campo usando el contorno del área verde.
    Más robusto que HoughLines cuando las líneas internas del campo (áreas de
    penalti, línea central) dominan sobre el borde exterior.

    Estrategia:
      1. Máscara HSV del color verde del campo.
      2. Morfología para obtener el blob más grande (sin huecos de robots/pelota).
      3. Envolvente convexa del blob → aproximar a cuadrilátero.
      4. Calcular homografía y validar con las porterías.
    """
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rango verde del campo — amplio para capturar distintos tipos de césped/alfombra
    GREEN_LO = np.array([35,  20,  30])
    GREEN_HI = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)

    # Cerrar huecos (robots, pelota, sombras) con kernel grande
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        if verbose:
            print("  Intento B: no se encontró área verde")
        return None, None

    # Tomar el contorno más grande
    field_cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(field_cnt) < 50000:
        if verbose:
            print("  Intento B: área verde demasiado pequeña")
        return None, None

    # Envolvente convexa → cuadrilátero
    hull = cv2.convexHull(field_cnt).reshape(-1, 2).astype(np.float32)

    # Aproximar a 4 puntos usando Douglas-Peucker con epsilon adaptativo
    peri = cv2.arcLength(hull.reshape(-1, 1, 2), True)
    for eps_frac in [0.02, 0.04, 0.06, 0.10]:
        approx = cv2.approxPolyDP(hull.reshape(-1, 1, 2), eps_frac * peri, True)
        if len(approx) <= 6:
            break

    pts = approx.reshape(-1, 2).astype(np.float32)

    # Si tiene más de 4 puntos, quedarse con los 4 del hull convexo más extremos
    if len(pts) != 4:
        sums = pts[:, 0] + pts[:, 1]
        difs = pts[:, 0] - pts[:, 1]
        idx4 = sorted(set([int(np.argmin(sums)), int(np.argmax(sums)),
                           int(np.argmax(difs)), int(np.argmin(difs))]))
        if len(idx4) < 4:
            if verbose:
                print(f"  Intento B: no se pudo reducir a 4 puntos ({len(pts)} puntos)")
            return None, None
        pts = pts[idx4]

    tl, tr, br, bl = _order_corners(pts)
    src = np.float32([tl, tr, br, bl])

    if cv2.contourArea(src) < 10000:
        if verbose:
            print("  Intento B: cuadrilátero demasiado pequeño")
        return None, None

    dst = np.float32([[0, 0], [FIELD_W, 0], [FIELD_W, FIELD_H], [0, FIELD_H]])
    Hm, _ = cv2.findHomography(src, dst)
    if Hm is None:
        return None, None

    # Validar con porterías si están disponibles
    if goal_y and goal_b and verbose:
        def _proj(pt):
            return cv2.perspectiveTransform(np.float32([[[pt[0], pt[1]]]]), Hm)[0][0]
        py, pb = _proj(goal_y), _proj(goal_b)
        print(f"  Intento B: portería_y→canvas={py.astype(int)}  portería_b→canvas={pb.astype(int)}")

    corners = {k: v.tolist() for k, v in
               zip(["top_left", "top_right", "bottom_right", "bottom_left"],
                   [tl, tr, br, bl])}
    if verbose:
        print(f"  Intento B OK — {len(approx)} puntos aproximados → cuadrilátero")
    return corners, Hm


# ── Intento C: estimación geométrica desde porterías ──────────────────────────

def _corners_from_goals_only(frame, goal_y, goal_b, verbose=True):
    """
    Último recurso cuando A y B fallan pero sí se detectaron las dos porterías.
    Las porterías están en los extremos izquierdo/derecho del campo a media altura.
    Usa ese ancla para estimar las cuatro esquinas del campo.
    """
    if not goal_y or not goal_b:
        return None, None

    H, W = frame.shape[:2]

    # Portería izquierda y derecha
    if goal_y[0] <= goal_b[0]:
        left_g, right_g = goal_y, goal_b
    else:
        left_g, right_g = goal_b, goal_y

    # La línea horizontal de las porterías define la mitad vertical del campo.
    # Estimamos que el campo se extiende la misma distancia hacia arriba y abajo.
    goal_avg_y = (left_g[1] + right_g[1]) / 2
    half_h     = goal_avg_y * 0.85          # ~85% de la distancia goal→top como mitad campo

    top_y = max(0.0, goal_avg_y - half_h)
    bot_y = min(float(H), goal_avg_y + half_h * 1.1)

    # Extensión horizontal: un poco más allá de las porterías
    margin_x = (right_g[0] - left_g[0]) * 0.05
    left_x  = max(0.0, left_g[0]  - margin_x)
    right_x = min(float(W), right_g[0] + margin_x)

    tl = [left_x,  top_y]
    tr = [right_x, top_y]
    br = [right_x, bot_y]
    bl = [left_x,  bot_y]

    corners = {"top_left": tl, "top_right": tr, "bottom_right": br, "bottom_left": bl}
    src = np.float32([tl, tr, br, bl])
    dst = np.float32([[0, 0], [FIELD_W, 0], [FIELD_W, FIELD_H], [0, FIELD_H]])
    Hm, _ = cv2.findHomography(src, dst)
    if Hm is None:
        return None, None

    if verbose:
        print("  Intento C OK — estimación geométrica desde porterías")
    return corners, Hm


# ── Intento A: líneas blancas exteriores ───────────────────────────────────────

def _corners_from_outer_white_lines(frame, goal_y, goal_b, verbose=True):
    """
    Busca las 4 líneas blancas más exteriores del campo (borde del rectángulo).

    Estrategia:
    1. Máscara verde del campo → dilatar para incluir las líneas del borde.
    2. Máscara blanca restringida a esa zona (evita líneas de paredes/marcadores).
    3. HoughLines con umbral bajo (líneas gruesas son fáciles de detectar).
    4. Clasificar como horizontal/vertical según ángulo.
    5. Elegir la más exterior en cada dirección → 4 líneas → 4 esquinas.
    """
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Campo verde — rango amplio
    GREEN_LO = np.array([35, 20, 30])
    GREEN_HI = np.array([100, 255, 255])
    green = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    k_g   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    green_adj = cv2.dilate(green, k_g)   # dilatar para incluir líneas blancas del borde

    # Máscara blanca restringida a zona verde-adyacente
    white = cv2.inRange(hsv, HSV_WHITE_LO, HSV_WHITE_HI)
    k_w   = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k_w)
    white_field = cv2.bitwise_and(white, green_adj)

    # Fallback: si no hay suficiente blanco en el campo, usar máscara global
    if cv2.countNonZero(white_field) < 300:
        if verbose:
            print("  Intento A: pocos píxeles blancos en campo → usando máscara global")
        white_field = white

    lines = cv2.HoughLines(white_field, 1, np.pi / 180, threshold=60)
    if lines is None or len(lines) < 4:
        if verbose:
            n = len(lines) if lines is not None else 0
            print(f"  Intento A: pocas líneas detectadas ({n})")
        return None, None

    if verbose:
        print(f"  Intento A: {len(lines)} líneas detectadas en zona blanca-verde")

    # Clasificar en horizontales / verticales y calcular posición real en el frame
    # Ecuación de la línea: x·cos(θ) + y·sin(θ) = ρ
    # — θ ≈ 0 o π  → línea casi vertical   → posición en x al centro vertical del frame
    # — θ ≈ π/2    → línea casi horizontal  → posición en y al centro horizontal del frame
    ANGLE_TOL = 0.40   # radianes (~23°)
    verticals   = []   # (x_at_center, rho, theta)
    horizontals = []   # (y_at_center, rho, theta)

    for r, t in lines[:, 0]:
        t = float(t) % np.pi
        r = float(r)
        if t < ANGLE_TOL or t > np.pi - ANGLE_TOL:
            cos_t = np.cos(t)
            if abs(cos_t) > 0.1:
                x_mid = (r - (H / 2) * np.sin(t)) / cos_t
                verticals.append((x_mid, r, t))
        elif abs(t - np.pi / 2) < ANGLE_TOL:
            sin_t = np.sin(t)
            if abs(sin_t) > 0.1:
                y_mid = (r - (W / 2) * np.cos(t)) / sin_t
                horizontals.append((y_mid, r, t))

    if len(verticals) < 2 or len(horizontals) < 2:
        if verbose:
            print(f"  Intento A: no suficientes líneas clasificadas "
                  f"({len(verticals)} vert, {len(horizontals)} horiz)")
        return None, None

    # Líneas más exteriores en cada dirección
    left_v  = min(verticals,   key=lambda v: v[0])
    right_v = max(verticals,   key=lambda v: v[0])
    top_h   = min(horizontals, key=lambda v: v[0])
    bot_h   = max(horizontals, key=lambda v: v[0])

    if verbose:
        print(f"  Intento A: izq_x≈{left_v[0]:.0f}  der_x≈{right_v[0]:.0f}  "
              f"top_y≈{top_h[0]:.0f}  bot_y≈{bot_h[0]:.0f}")

    tl = _intersect(left_v[1],  left_v[2],  top_h[1], top_h[2])
    tr = _intersect(right_v[1], right_v[2], top_h[1], top_h[2])
    br = _intersect(right_v[1], right_v[2], bot_h[1], bot_h[2])
    bl = _intersect(left_v[1],  left_v[2],  bot_h[1], bot_h[2])

    if not all([tl, tr, br, bl]):
        if verbose:
            print("  Intento A: no se pudieron calcular intersecciones")
        return None, None

    src = np.float32([tl, tr, br, bl])
    if cv2.contourArea(src) < 5000:
        if verbose:
            print("  Intento A: cuadrilátero demasiado pequeño")
        return None, None

    dst   = np.float32([[0, 0], [FIELD_W, 0], [FIELD_W, FIELD_H], [0, FIELD_H]])
    Hm, _ = cv2.findHomography(src, dst)
    if Hm is None:
        return None, None

    if goal_y and goal_b and verbose:
        def _proj(pt):
            return cv2.perspectiveTransform(np.float32([[[pt[0], pt[1]]]]), Hm)[0][0]
        py, pb = _proj(goal_y), _proj(goal_b)
        print(f"  Intento A: portería_y→canvas={py.astype(int)}  portería_b→canvas={pb.astype(int)}")

    corners = {
        "top_left":     list(tl), "top_right":    list(tr),
        "bottom_right": list(br), "bottom_left":  list(bl),
    }
    if verbose:
        print("  Intento A OK — líneas blancas exteriores")
    return corners, Hm


# ── Detección de corners ────────────────────────────────────────────────────────

def find_corners(frame, verbose=True):
    H, W = frame.shape[:2]

    goal_y = _detect_goal(frame, HSV_YELLOW_LO, HSV_YELLOW_HI)
    goal_b = _detect_goal(frame, HSV_BLUE_LO,   HSV_BLUE_HI)
    if verbose:
        print(f"  Portería amarilla: {goal_y}")
        print(f"  Portería azul    : {goal_b}")

    # ── Intento A: líneas blancas exteriores ──────────────────────────────────
    best_corners, best_H = _corners_from_outer_white_lines(frame, goal_y, goal_b, verbose)

    # ── Intento B: silueta del área verde ────────────────────────────────────
    if best_corners is None:
        if verbose:
            print("  Intento A falló → Intento B: silueta del área verde")
        best_corners, best_H = _corners_from_green_hull(frame, goal_y, goal_b, verbose)

    # ── Intento C: estimación geométrica desde porterías ─────────────────────
    if best_corners is None:
        if verbose:
            print("  Intento B falló → Intento C: estimación desde porterías")
        best_corners, best_H = _corners_from_goals_only(frame, goal_y, goal_b, verbose)

    if best_corners is None:
        print("  ERROR: no se pudo determinar las esquinas.")
        return None, None, (goal_y, goal_b)

    # Anclar al borde del frame las esquinas que caen dentro de la tolerancia
    best_corners = _snap_corners(best_corners, W, H)

    # Recalcular homografía con las esquinas definitivas (post-snap)
    order = ["top_left", "top_right", "bottom_right", "bottom_left"]
    src   = np.float32([best_corners[k] for k in order])
    dst   = np.float32([[0, 0], [FIELD_W, 0], [FIELD_W, FIELD_H], [0, FIELD_H]])
    best_H, _ = cv2.findHomography(src, dst)

    return best_corners, best_H, (goal_y, goal_b)


# ── Cuadrícula de coordenadas (helper para selección manual) ───────────────────

def draw_coord_grid(frame, out_path, step=100):
    """
    Genera una imagen con cuadrícula de coordenadas cada STEP píxeles.
    Útil para identificar visualmente las esquinas del campo.
    Abre en Windows con:  explorer.exe "$(wslpath -w <ruta>)"
    """
    H, W = frame.shape[:2]
    vis  = frame.copy()

    for x in range(0, W + 1, step):
        color = (0, 200, 200) if x % 500 == 0 else (80, 80, 80)
        thick = 2              if x % 500 == 0 else 1
        cv2.line(vis, (x, 0), (x, H), color, thick)
        if x % 200 == 0 and x > 0:
            cv2.putText(vis, str(x), (x + 3, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 2, cv2.LINE_AA)

    for y in range(0, H + 1, step):
        color = (0, 200, 200) if y % 500 == 0 else (80, 80, 80)
        thick = 2              if y % 500 == 0 else 1
        cv2.line(vis, (0, y), (W, y), color, thick)
        if y % 200 == 0 and y > 0:
            cv2.putText(vis, str(y), (6, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 2, cv2.LINE_AA)

    # Etiqueta de origen
    cv2.putText(vis, "0,0", (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, vis)
    abs_path = os.path.abspath(out_path)
    print(f"  Cuadricula: {out_path}")
    print(f"  Ver en Windows: explorer.exe \"$(wslpath -w {abs_path})\"")


# ── Imagen de debug ────────────────────────────────────────────────────────────

def draw_debug(frame, corners, H_mat, goals, out_path):
    vis = frame.copy()
    H, W = vis.shape[:2]

    if corners:
        order  = ["top_left", "top_right", "bottom_right", "bottom_left"]
        colors = {"top_left":(0,0,255), "top_right":(0,255,255),
                  "bottom_right":(255,0,0), "bottom_left":(255,255,0)}
        poly   = []
        for name in order:
            pt = corners[name]
            poly.append([int(np.clip(pt[0],0,W-1)), int(np.clip(pt[1],0,H-1))])
        cv2.polylines(vis, [np.array(poly, dtype=np.int32)], True, (0,255,0), 3)
        for name in order:
            pt = corners[name]
            px, py = int(np.clip(pt[0],5,W-5)), int(np.clip(pt[1],5,H-5))
            in_f = (0 <= pt[0] <= W and 0 <= pt[1] <= H)
            cv2.circle(vis, (px, py), 12, colors[name], -1)
            cv2.circle(vis, (px, py), 14, (255,255,255), 2)
            cv2.putText(vis, f"{name}" + ("" if in_f else " (extrapolado)"),
                        (px+15, py+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colors[name], 2, cv2.LINE_AA)

    goal_y, goal_b = goals
    for pt, label, color in [(goal_y, "YELLOW", (0,220,255)),
                              (goal_b, "BLUE",   (255,100,0))]:
        if pt:
            p = (int(pt[0]), int(pt[1]))
            cv2.circle(vis, p, 15, color, -1)
            cv2.putText(vis, label, (p[0]+18, p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Mini vista cenital como inset
    if H_mat is not None:
        fw, fh = FIELD_W // 3, FIELD_H // 3
        td = np.full((fh, fw, 3), (30, 120, 30), dtype=np.uint8)
        cv2.rectangle(td, (3,3), (fw-3,fh-3), (255,255,255), 1)
        cv2.line(td, (fw//2,3), (fw//2,fh-3), (255,255,255), 1)
        def _proj_td(cam_pt):
            p = cv2.perspectiveTransform(
                np.float32([[[cam_pt[0], cam_pt[1]]]]), H_mat)[0][0]
            return (int(np.clip(p[0]*fw/FIELD_W, 0, fw-1)),
                    int(np.clip(p[1]*fh/FIELD_H, 0, fh-1)))
        if goal_y:
            cv2.circle(td, _proj_td(goal_y), 6, (0,220,255), -1)
        if goal_b:
            cv2.circle(td, _proj_td(goal_b), 6, (255,100,0), -1)
        vis[10:10+fh, 10:10+fw] = td
        cv2.rectangle(vis, (10,10), (10+fw,10+fh), (200,200,200), 2)
        cv2.putText(vis, "Vista cenital (preview)", (10, 10+fh+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f"  Debug: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_pt(s, name):
    try:
        x, y = s.split(",")
        return [float(x.strip()), float(y.strip())]
    except Exception:
        print(f"ERROR: formato inválido para {name}: '{s}'  (usa x,y  ej: 208,620)")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame", required=True,
                        help="Frame del video (JPG/PNG)")
    parser.add_argument("--out",   default="output/field_corners.json",
                        help="JSON de salida")
    parser.add_argument("--debug", default="",
                        help="Imagen de debug (default: <out>_debug.jpg)")
    parser.add_argument("--grid",  default="",
                        help="Genera imagen con cuadrícula de coordenadas para selección manual")
    parser.add_argument("--tl", default="", metavar="x,y",
                        help="Top-left  manual (modo sin detección automática)")
    parser.add_argument("--tr", default="", metavar="x,y",
                        help="Top-right manual")
    parser.add_argument("--br", default="", metavar="x,y",
                        help="Bottom-right manual")
    parser.add_argument("--bl", default="", metavar="x,y",
                        help="Bottom-left manual")
    args = parser.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        print(f"ERROR: no se puede leer {args.frame}"); sys.exit(1)

    H, W = frame.shape[:2]
    print(f"Frame: {args.frame}  ({W}×{H})")

    # ── Cuadrícula de coordenadas (solo genera imagen, no modifica JSON) ──────
    if args.grid:
        draw_coord_grid(frame, args.grid)
        if not any([args.tl, args.tr, args.br, args.bl]):
            print("\nAbre la imagen de cuadricula en Windows, identifica las 4 esquinas")
            print("del campo y vuelve a ejecutar con:")
            print(f"  python scripts/auto_corners.py --frame {args.frame} \\")
            print(f"      --tl X,Y --tr X,Y --br X,Y --bl X,Y \\")
            print(f"      --out {args.out}")
            return

    # ── Modo manual: 4 esquinas especificadas por CLI ─────────────────────────
    manual_pts = [args.tl, args.tr, args.br, args.bl]
    if any(manual_pts):
        missing = [n for n, v in zip(["--tl","--tr","--br","--bl"], manual_pts) if not v]
        if missing:
            print(f"ERROR: modo manual requiere los 4 puntos. Falta: {missing}")
            sys.exit(1)

        corners = {
            "top_left":     _parse_pt(args.tl, "--tl"),
            "top_right":    _parse_pt(args.tr, "--tr"),
            "bottom_right": _parse_pt(args.br, "--br"),
            "bottom_left":  _parse_pt(args.bl, "--bl"),
        }
        corners  = _snap_corners(corners, W, H)
        order    = ["top_left", "top_right", "bottom_right", "bottom_left"]
        src      = np.float32([corners[k] for k in order])
        dst      = np.float32([[0,0],[FIELD_W,0],[FIELD_W,FIELD_H],[0,FIELD_H]])
        H_mat, _ = cv2.findHomography(src, dst)
        goals    = (_detect_goal(frame, HSV_YELLOW_LO, HSV_YELLOW_HI),
                    _detect_goal(frame, HSV_BLUE_LO,   HSV_BLUE_HI))
        print("Modo manual — esquinas especificadas por CLI")

    # ── Modo automático ───────────────────────────────────────────────────────
    else:
        corners, H_mat, goals = find_corners(frame, verbose=True)
        if corners is None:
            abs_frame = os.path.abspath(args.frame)
            grid_path = args.out.replace(".json", "_grid.jpg")
            print(f"\nDeteccion automatica fallida.")
            print(f"\nPaso 1 — genera la cuadricula de coordenadas:")
            print(f"  python scripts/auto_corners.py --frame {args.frame} --grid {grid_path}")
            print(f"  explorer.exe \"$(wslpath -w {os.path.abspath(grid_path)})\"")
            print(f"\nPaso 2 — especifica las esquinas manualmente:")
            print(f"  python scripts/auto_corners.py --frame {args.frame} \\")
            print(f"      --tl X,Y --tr X,Y --br X,Y --bl X,Y \\")
            print(f"      --out {args.out}")
            sys.exit(1)

    # ── Imprimir y guardar resultado ──────────────────────────────────────────
    print("\nEsquinas:")
    for name, pt in corners.items():
        in_f = (0 <= pt[0] <= W and 0 <= pt[1] <= H)
        print(f"  {name:14s}: ({int(pt[0]):6d}, {int(pt[1]):6d})"
              f"  {'✓' if in_f else '⚠ extrapolado'}")

    if H_mat is not None and goals[0] and goals[1]:
        def _proj(pt):
            return cv2.perspectiveTransform(np.float32([[[pt[0],pt[1]]]]), H_mat)[0][0]
        py, pb = _proj(goals[0]), _proj(goals[1])
        print(f"\nValidacion porterias → canvas cenital:")
        print(f"  Amarilla: ({int(py[0])}, {int(py[1])})")
        print(f"  Azul:     ({int(pb[0])}, {int(pb[1])})")

    result = {
        "frame":         args.frame,
        "frame_size":    [W, H],
        "corners_image": {k: [int(v[0]), int(v[1])] for k, v in corners.items()},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nGuardado: {args.out}")

    debug = args.debug or args.out.replace(".json", "_debug.jpg")
    draw_debug(frame, corners, H_mat, goals, debug)


if __name__ == "__main__":
    main()
