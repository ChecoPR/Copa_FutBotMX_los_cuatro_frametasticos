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
from itertools import combinations

import cv2
import numpy as np

# ── Parámetros ─────────────────────────────────────────────────────────────────
HSV_WHITE_LO  = np.array([0,   0, 155])
HSV_WHITE_HI  = np.array([180, 60, 255])
HSV_YELLOW_LO = np.array([15, 100, 100])
HSV_YELLOW_HI = np.array([38, 255, 255])
HSV_BLUE_LO   = np.array([95, 100,  40])
HSV_BLUE_HI   = np.array([130, 255, 255])

HOUGH_THRESH = 100
MORPH_K      = 8
ANGLE_TOL    = 0.20
N_CANDS      = 6
FIELD_W      = 800
FIELD_H      = 540


# ── Utilidades ─────────────────────────────────────────────────────────────────

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


def _cluster_lines(lines, tol=ANGLE_TOL):
    rts = []
    for r, t in lines[:, 0]:
        t = float(t) % np.pi
        r = float(r)
        rts.append((r, t))
    clusters = []
    for r, t in sorted(rts, key=lambda x: x[1]):
        for cl in clusters:
            if abs(cl[1][-1] - t) < tol:
                cl[0].append(r); cl[1].append(t); break
        else:
            clusters.append(([r], [t]))
    return sorted([(np.mean(rs), np.mean(ts), len(rs))
                   for rs, ts in clusters], key=lambda x: -x[2])


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

    # Rango verde del campo (más amplio que el HSV_FIELD de auto_detect)
    GREEN_LO = np.array([40,  30,  40])
    GREEN_HI = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)

    # Cerrar huecos (robots, pelota, sombras)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
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


# ── Detección de corners ────────────────────────────────────────────────────────

def find_corners(frame, verbose=True):
    H, W = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    goal_y = _detect_goal(frame, HSV_YELLOW_LO, HSV_YELLOW_HI)
    goal_b = _detect_goal(frame, HSV_BLUE_LO,   HSV_BLUE_HI)
    if verbose:
        print(f"  Portería amarilla: {goal_y}")
        print(f"  Portería azul    : {goal_b}")

    white = cv2.inRange(hsv, HSV_WHITE_LO, HSV_WHITE_HI)
    k     = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)
    edges = cv2.Canny(white, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=HOUGH_THRESH)

    clusters = []
    if lines is not None and len(lines) >= 4:
        clusters = _cluster_lines(lines)
        if verbose:
            print(f"  {len(lines)} líneas → {len(clusters)} clusters:")
            for r, t, n in clusters[:8]:
                print(f"    rho={r:7.1f}  theta={np.degrees(t):6.1f}°  n={n}")

    best_score, best_corners, best_H = -1e9, None, None

    # ── Intento A: combinaciones de 4 líneas ──────────────────────────────────
    top_n = clusters[:N_CANDS]
    for four in combinations(range(len(top_n)), 4):
        lines4 = [top_n[i] for i in four]
        angles = [l[1] for l in lines4]

        # Descartar si hay líneas casi paralelas (diff < 15°)
        skip = any(
            np.degrees(min(abs(angles[i] - angles[j]),
                           np.pi - abs(angles[i] - angles[j]))) < 15.0
            for i in range(4) for j in range(i + 1, 4)
        )
        if skip:
            continue

        pts = [p for i in range(4) for j in range(i + 1, 4)
               if (p := _intersect(lines4[i][0], lines4[i][1],
                                   lines4[j][0], lines4[j][1]))]
        if len(pts) < 4:
            continue

        arr  = np.array(pts, dtype=np.float32)
        hull = cv2.convexHull(arr).reshape(-1, 2)
        if len(hull) < 4:
            continue

        sums, difs = hull[:, 0] + hull[:, 1], hull[:, 0] - hull[:, 1]
        idxs = sorted(set(map(int, [np.argmin(sums), np.argmax(sums),
                                    np.argmax(difs),  np.argmin(difs)])))
        if len(idxs) < 4:
            continue
        hull = hull[idxs]

        tl, tr, br, bl = _order_corners(hull)
        src = np.float32([tl, tr, br, bl])
        if cv2.contourArea(src) < 5000:
            continue

        dst   = np.float32([[0,0],[FIELD_W,0],[FIELD_W,FIELD_H],[0,FIELD_H]])
        Hm, _ = cv2.findHomography(src, dst)
        if Hm is None:
            continue

        if goal_y and goal_b:
            def _proj(pt, Hm=Hm):
                return cv2.perspectiveTransform(
                    np.float32([[[pt[0], pt[1]]]]), Hm)[0][0]
            py, pb = _proj(goal_y), _proj(goal_b)
            M  = 300
            ok = (-M < py[0] < FIELD_W+M and -M < py[1] < FIELD_H+M and
                  -M < pb[0] < FIELD_W+M and -M < pb[1] < FIELD_H+M)
            dx = float(py[0]) - float(pb[0])
            score = dx if (ok and dx > 0) else -1e9
        else:
            score = cv2.contourArea(src)

        if score > best_score:
            best_score = score
            best_corners = {k: v.tolist() for k, v in
                            zip(["top_left","top_right","bottom_right","bottom_left"],
                                [tl, tr, br, bl])}
            best_H = Hm

    # ── Intento B: silueta del área verde (contorno del campo) ───────────────────
    if best_corners is None:
        if verbose:
            print("  Intento A falló → Intento B: silueta del área verde")
        best_corners, best_H = _corners_from_green_hull(frame, goal_y, goal_b, verbose)

    if best_corners is None:
        print("  ERROR: no se pudo determinar las esquinas.")

    return best_corners, best_H, (goal_y, goal_b)


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

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--out",   default="output/field_corners.json")
    parser.add_argument("--debug", default="")
    args = parser.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        print(f"ERROR: no se puede leer {args.frame}"); sys.exit(1)

    H, W = frame.shape[:2]
    print(f"Frame: {args.frame}  ({W}×{H})")

    corners, H_mat, goals = find_corners(frame, verbose=True)
    if corners is None:
        sys.exit(1)

    print("\nEsquinas:")
    for name, pt in corners.items():
        in_f = (0 <= pt[0] <= W and 0 <= pt[1] <= H)
        print(f"  {name:14s}: ({int(pt[0]):6d}, {int(pt[1]):6d})"
              f"  {'✓' if in_f else '⚠ extrapolado'}")

    result = {
        "frame":         args.frame,
        "frame_size":    [W, H],
        "corners_image": {k: [int(v[0]), int(v[1])] for k, v in corners.items()},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Guardado: {args.out}")

    debug = args.debug or args.out.replace(".json", "_debug.jpg")
    draw_debug(frame, corners, H_mat, goals, debug)


if __name__ == "__main__":
    main()
