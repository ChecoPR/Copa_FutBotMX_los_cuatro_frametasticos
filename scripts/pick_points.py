"""
Herramienta interactiva para seleccionar puntos en el frame 0.

Uso:
    python scripts/pick_points.py --frames_dir output/frames/IMG_9866

Controles:
    Clic izquierdo  — marca un punto (te pedirá la etiqueta en terminal)
    Z               — deshacer último punto
    Enter / Q       — terminar y mostrar el comando listo para copiar
"""

import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

COLORS = [
    "#00ff00", "#00c8ff", "#ff6400",
    "#ff00c8", "#0064ff", "#c8ff00",
]

points = []   # lista de (label, x, y)
fig = ax = base_img_rgb = None


def redraw():
    ax.clear()
    ax.imshow(base_img_rgb)
    ax.set_title("Clic = agregar punto   Z = deshacer   Enter/Q = terminar")
    ax.axis("off")
    for i, (label, x, y) in enumerate(points):
        c = COLORS[i % len(COLORS)]
        ax.plot(x, y, "o", color=c, markersize=10, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(x + 18, y + 10, f"{i+1}:{label}", color=c, fontsize=11,
                fontweight="bold", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0))
    fig.canvas.draw()


def on_click(event):
    if event.inaxes != ax or event.button != 1:
        return
    x, y = int(round(event.xdata)), int(round(event.ydata))
    label = input(f"  Etiqueta para punto ({x},{y}): ").strip()
    if label:
        points.append((label, x, y))
        print(f"  -> '{label}' en ({x}, {y})")
        redraw()


def on_key(event):
    if event.key in ("enter", "q", "Q"):
        plt.close()
    elif event.key in ("z", "Z"):
        if points:
            removed = points.pop()
            print(f"  Deshecho: '{removed[0]}' en ({removed[1]}, {removed[2]})")
            redraw()


def main():
    global fig, ax, base_img_rgb

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--frame_index", type=int, default=0)
    args = parser.parse_args()

    exts = {".jpg", ".jpeg", ".png"}
    frame_files = sorted([
        f for f in os.listdir(args.frames_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    if not frame_files:
        print(f"No hay frames en {args.frames_dir}")
        sys.exit(1)

    idx = min(args.frame_index, len(frame_files) - 1)
    frame_path = os.path.join(args.frames_dir, frame_files[idx])
    bgr = cv2.imread(frame_path)
    h, w = bgr.shape[:2]
    base_img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print(f"Frame: {frame_files[idx]}  ({w}x{h})")
    print("Haz clic sobre cada objeto. Escribe la etiqueta en terminal.")
    print("  Z = deshacer  |  Enter o Q = terminar\n")

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.tight_layout(pad=0.5)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()

    if not points:
        print("No se seleccionó ningún punto.")
        sys.exit(0)

    print("\n=== Puntos seleccionados ===")
    for label, x, y in points:
        print(f"  {label}: ({x}, {y})")

    point_str = " ".join(f"{label}:{x},{y}" for label, x, y in points)
    print("\n=== Comando para copiar y pegar ===")
    print(f'python scripts/pipeline.py --frames_dir {args.frames_dir} --point_prompts "{point_str}"')


if __name__ == "__main__":
    main()
