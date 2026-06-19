#!/usr/bin/env python3
"""
Selección interactiva de las 4 esquinas del campo para la homografía.

Instrucciones:
  1. Se abre el primer frame.
  2. Haz clic en las 4 esquinas del campo EN ESTE ORDEN:
       Superior-Izquierda → Superior-Derecha → Inferior-Derecha → Inferior-Izquierda
  3. Presiona ENTER para confirmar, ESC para borrar el último punto.
  4. Las coordenadas se guardan en output/field_corners.json.

Uso:
    python scripts/pick_field_corners.py --frame output/frames/IMG_9866/00000.jpg
    python scripts/pick_field_corners.py --frame output/frames/IMG_9866/00000.jpg \
        --out output/field_corners_IMG_9866.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

CORNER_NAMES = [
    "Superior-Izquierda (top-left)",
    "Superior-Derecha   (top-right)",
    "Inferior-Derecha   (bottom-right)",
    "Inferior-Izquierda (bottom-left)",
]
COLORS = [(0, 200, 255), (0, 255, 100), (0, 80, 255), (255, 130, 0)]

clicks = []
img_orig = None


def _draw(img):
    vis = img.copy()
    n = len(clicks)
    for i, (x, y) in enumerate(clicks):
        color = COLORS[i]
        cv2.circle(vis, (x, y), 8, color, -1)
        cv2.circle(vis, (x, y), 10, (255, 255, 255), 2)
        cv2.putText(vis, f"{i+1}: {CORNER_NAMES[i].split('(')[0].strip()}",
                    (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    if n >= 2:
        for i in range(n - 1):
            cv2.line(vis, clicks[i], clicks[i+1], (200, 200, 200), 1, cv2.LINE_AA)
    if n == 4:
        cv2.line(vis, clicks[3], clicks[0], (200, 200, 200), 1, cv2.LINE_AA)
        poly = np.array(clicks, dtype=np.int32)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [poly], (0, 255, 0))
        cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)

    # Instrucciones
    remaining = CORNER_NAMES[n] if n < 4 else "ENTER para guardar, ESC para borrar último"
    cv2.putText(vis, f"Siguiente: {remaining}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"Puntos: {n}/4  |  ESC=borrar último  |  ENTER=guardar",
                (15, vis.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return vis


def _on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
        clicks.append((x, y))
        cv2.imshow("Seleccionar esquinas", _draw(img_orig))


def main():
    global img_orig

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame", required=True, help="Frame de referencia (jpg/png)")
    parser.add_argument("--out",   default="output/field_corners.json",
                        help="Archivo JSON de salida")
    args = parser.parse_args()

    img_orig = cv2.imread(args.frame)
    if img_orig is None:
        print(f"No se pudo leer: {args.frame}"); sys.exit(1)

    H, W = img_orig.shape[:2]
    print(f"\nFrame: {args.frame}  ({W}×{H})")
    print("Haz clic en las 4 esquinas del CAMPO en orden:")
    for i, n in enumerate(CORNER_NAMES):
        print(f"  {i+1}. {n}")
    print("\nESC = borrar último punto   ENTER = guardar y salir")

    cv2.namedWindow("Seleccionar esquinas", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Seleccionar esquinas", min(W, 1400), min(H * 1400 // W, 900))
    cv2.setMouseCallback("Seleccionar esquinas", _on_click)
    cv2.imshow("Seleccionar esquinas", _draw(img_orig))

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC — borrar último
            if clicks:
                clicks.pop()
                cv2.imshow("Seleccionar esquinas", _draw(img_orig))
        elif key == 13 or key == 10:  # ENTER
            if len(clicks) < 4:
                print(f"Faltan {4 - len(clicks)} puntos.")
            else:
                break
        elif key == ord('q'):
            print("Cancelado."); cv2.destroyAllWindows(); sys.exit(0)

    cv2.destroyAllWindows()

    corners = {
        "frame":          args.frame,
        "frame_size":     [W, H],
        "corners_image":  {
            "top_left":     list(clicks[0]),
            "top_right":    list(clicks[1]),
            "bottom_right": list(clicks[2]),
            "bottom_left":  list(clicks[3]),
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(corners, f, indent=2)

    print(f"\nEsquinas guardadas en: {args.out}")
    for k, v in corners["corners_image"].items():
        print(f"  {k:15s}: {v}")


if __name__ == "__main__":
    main()
