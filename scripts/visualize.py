#!/usr/bin/env python3
"""
Visualizaciones para Copa FutBotMX.

Subcomandos:
  heatmap   — densidad de posiciones sobre el frame de cámara
  topdown   — vista cenital 2D del campo con heatmaps, trails y Voronoi
  trails    — trayectorias con degradado temporal (en frame de cámara)
  voronoi   — zonas de control por frame

Uso:
    python scripts/visualize.py heatmap \
        --analytics output/analytics/IMG_9866_analytics.json \
        --bg        output/frames/IMG_9866/00000.jpg \
        --output    output/viz/IMG_9866/

    python scripts/visualize.py topdown \
        --analytics output/analytics/IMG_9866_analytics.json \
        --corners   output/field_corners_IMG_9866.json \
        --output    output/viz/IMG_9866/

    python scripts/visualize.py trails  --analytics ... --bg ... --output ...
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


# ── Paleta de colores por label ────────────────────────────────────────────────
COLORS = {
    "robot1": (255,  80,   0),   # naranja
    "robot2": (  0,  80, 255),   # azul
    "ball":   (  0, 230,  80),   # verde brillante
}

CMAPS_BW = {
    "robot1": cv2.COLORMAP_HOT,
    "robot2": cv2.COLORMAP_COOL,
    "ball":   cv2.COLORMAP_SUMMER,
}


# ── Utilidades ─────────────────────────────────────────────────────────────────

def _gaussian_heatmap(points, h, w, sigma=40):
    """
    Genera un mapa de calor 2D (float32, rango 0-1) sumando kernels gaussianos
    centrados en cada punto.
    """
    canvas = np.zeros((h, w), dtype=np.float32)
    for cx, cy in points:
        cx, cy = int(round(cx)), int(round(cy))
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        # Ventana de ±3σ para eficiencia
        r = int(3 * sigma)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        gx = np.arange(x0, x1) - cx
        gy = np.arange(y0, y1) - cy
        gx2 = np.exp(-gx**2 / (2 * sigma**2))
        gy2 = np.exp(-gy**2 / (2 * sigma**2))
        canvas[y0:y1, x0:x1] += np.outer(gy2, gx2)
    if canvas.max() > 0:
        canvas /= canvas.max()
    return canvas


def _overlay_heatmap(bg, heatmap, cmap_id, alpha=0.55):
    """
    Mezcla el heatmap (float32 0-1) sobre bg (BGR uint8).
    Solo pinta píxeles con valor > 0.05 para no oscurecer el campo.
    """
    h8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(h8, cmap_id)   # BGR
    mask    = (heatmap > 0.05).astype(np.float32)
    out     = bg.copy().astype(np.float32)
    out     = out * (1 - mask[:, :, None] * alpha) + colored.astype(np.float32) * mask[:, :, None] * alpha
    return out.astype(np.uint8)


def _load_bg(bg_path, w=1920, h=1080):
    if bg_path and os.path.exists(bg_path):
        img = cv2.imread(bg_path)
        if img is not None:
            return img
    return np.full((h, w, 3), 40, dtype=np.uint8)


# ── Subcomandos ────────────────────────────────────────────────────────────────

def cmd_heatmap(args):
    with open(args.analytics) as f:
        data = json.load(f)

    paths   = data["paths"]         # {label: [[cx,cy],...]}
    summary = data["summary"]
    os.makedirs(args.output, exist_ok=True)

    bg_orig = _load_bg(args.bg)
    H, W    = bg_orig.shape[:2]

    labels = [l for l in paths if paths[l]]

    # ── Heatmap individual por objeto ──────────────────────────────────────
    individual = {}
    for lbl in labels:
        pts  = paths[lbl]
        hmap = _gaussian_heatmap(pts, H, W, sigma=args.sigma)
        individual[lbl] = hmap

        cmap  = CMAPS_BW.get(lbl, cv2.COLORMAP_HOT)
        vis   = _overlay_heatmap(bg_orig, hmap, cmap, alpha=0.6)

        # Leyenda
        color = COLORS.get(lbl, (200, 200, 200))
        poss  = summary["possession"].get(lbl, {})
        spd   = summary["speed_avg_px_s"].get(lbl, 0)
        txt   = (f"{lbl}  |  {poss.get('pct', 0):.1f}% posesion  "
                 f"|  {spd} px/s  |  {len(pts)} frames")
        cv2.putText(vis, txt, (20, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        out_path = os.path.join(args.output, f"heatmap_{lbl}.jpg")
        cv2.imwrite(out_path, vis)
        print(f"  → {out_path}")

    # ── Heatmap combinado (todos los objetos, colores distintos) ───────────
    combined_vis = bg_orig.copy().astype(np.float32)
    for lbl, hmap in individual.items():
        color = COLORS.get(lbl, (200, 200, 200))
        # Crear capa de color sólido para este objeto
        layer     = np.zeros((H, W, 3), dtype=np.float32)
        layer[:]  = color[::-1]           # BGR
        mask      = hmap[:, :, None]
        combined_vis = (combined_vis * (1 - mask * 0.5) +
                        layer          * mask * 0.5)

    combined_vis = combined_vis.clip(0, 255).astype(np.uint8)

    # Leyenda combinada
    y = 30
    for lbl in labels:
        color = COLORS.get(lbl, (200, 200, 200))
        poss  = summary["possession"].get(lbl, {})
        spd   = summary["speed_avg_px_s"].get(lbl, 0)
        txt   = f"{lbl}: {poss.get('pct',0):.1f}% pos  {spd} px/s"
        cv2.putText(combined_vis, txt, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 28

    out_path = os.path.join(args.output, "heatmap_combined.jpg")
    cv2.imwrite(out_path, combined_vis)
    print(f"  → {out_path}")

    # ── Panel 2×2 con matplotlib ───────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        robot_labels = [l for l in labels if l.startswith("robot")]
        n_robots = len(robot_labels)
        ncols = 2
        nrows = (n_robots + 2) // ncols   # robots + ball + combined

        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        axes = axes.flatten()

        bg_rgb = cv2.cvtColor(bg_orig, cv2.COLOR_BGR2RGB)

        panel_labels = robot_labels + ["ball"] + (["combined"] if len(labels) > 1 else [])
        for ax, lbl in zip(axes, panel_labels):
            ax.imshow(bg_rgb, alpha=0.6)
            if lbl == "combined":
                for sub_lbl, hmap in individual.items():
                    c = [x/255 for x in COLORS.get(sub_lbl, (200,200,200))]
                    rgba = np.zeros((H, W, 4), dtype=np.float32)
                    rgba[:,:,0] = c[0]; rgba[:,:,1] = c[1]; rgba[:,:,2] = c[2]
                    rgba[:,:,3] = hmap * 0.7
                    ax.imshow(rgba)
                ax.set_title("Combinado", fontsize=11)
            else:
                hmap = individual.get(lbl)
                if hmap is not None:
                    c = [x/255 for x in COLORS.get(lbl, (200,200,200))]
                    rgba = np.zeros((H, W, 4), dtype=np.float32)
                    rgba[:,:,0]=c[0]; rgba[:,:,1]=c[1]; rgba[:,:,2]=c[2]
                    rgba[:,:,3] = hmap * 0.75
                    ax.imshow(rgba)
                    poss = summary["possession"].get(lbl, {})
                    spd  = summary["speed_avg_px_s"].get(lbl, 0)
                    ax.set_title(
                        f"{lbl}  |  {poss.get('pct',0):.1f}% posesión  |  {spd} px/s",
                        fontsize=11)
            ax.axis("off")

        # Ocultar subplots sin contenido
        for ax in axes[len(panel_labels):]:
            ax.set_visible(False)

        fig.suptitle("Heatmaps de actividad — Copa FutBotMX", fontsize=14, fontweight="bold")
        plt.tight_layout()
        panel_path = os.path.join(args.output, "heatmap_panel.png")
        fig.savefig(panel_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {panel_path}")
    except ImportError:
        print("  (matplotlib no disponible — se omite panel PNG)")


def _build_homography(corners_path, field_w=800, field_h=540):
    """
    Lee las 4 esquinas del campo (JSON) y devuelve la matriz de homografía
    imagen→campo y el tamaño del canvas cenital (field_w, field_h).
    """
    with open(corners_path) as f:
        c = json.load(f)["corners_image"]

    src = np.array([
        c["top_left"], c["top_right"],
        c["bottom_right"], c["bottom_left"],
    ], dtype=np.float32)

    dst = np.array([
        [0,       0      ],
        [field_w, 0      ],
        [field_w, field_h],
        [0,       field_h],
    ], dtype=np.float32)

    H_mat, _ = cv2.findHomography(src, dst)
    return H_mat, field_w, field_h


def _transform_pts(pts, H_mat):
    """Proyecta lista de [cx,cy] con la homografía. Retorna array (N,2)."""
    if not pts:
        return np.empty((0, 2))
    arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(arr, H_mat)
    return out.reshape(-1, 2)


def _draw_field_2d(W, H, line_color=(255, 255, 255)):
    """Canvas verde con líneas de campo: borde, línea central y círculo."""
    canvas = np.full((H, W, 3), (30, 120, 30), dtype=np.uint8)
    pad = 15
    lw  = 2

    # Borde del campo
    cv2.rectangle(canvas, (pad, pad), (W - pad, H - pad), line_color, lw)

    # Línea de centro vertical
    cx = W // 2
    cv2.line(canvas, (cx, pad), (cx, H - pad), line_color, lw)

    # Círculo central
    r = min(W, H) // 8
    cv2.circle(canvas, (cx, H // 2), r, line_color, lw)
    cv2.circle(canvas, (cx, H // 2), 4, line_color, -1)

    # Áreas de gol (izquierda y derecha)
    gw = W // 8          # profundidad del área
    gh = H // 3          # alto del área
    gy1 = (H - gh) // 2
    gy2 = gy1 + gh
    cv2.rectangle(canvas, (pad,       gy1), (pad + gw,       gy2), line_color, lw)
    cv2.rectangle(canvas, (W-pad-gw,  gy1), (W - pad,        gy2), line_color, lw)

    return canvas


def cmd_topdown(args):
    """
    Vista cenital 2D del campo con:
      · Heatmap de densidad de posiciones (gaussian) por objeto
      · Trails con degradado temporal
      · Zonas de Voronoi (control territorial)
      · Panel con estadísticas de posesión y velocidad
    """
    with open(args.analytics) as f:
        data = json.load(f)

    paths   = data["paths"]
    summary = data["summary"]
    os.makedirs(args.output, exist_ok=True)

    # ── Homografía ──────────────────────────────────────────────────────────
    FW, FH = 800, 540
    H_mat, FW, FH = _build_homography(args.corners, FW, FH)

    # Transformar todas las trayectorias al espacio cenital
    td_paths = {}
    for lbl, pts in paths.items():
        td = _transform_pts(pts, H_mat)
        # Filtrar puntos fuera del canvas (proyecciones erróneas)
        valid = (td[:, 0] >= 0) & (td[:, 0] < FW) & (td[:, 1] >= 0) & (td[:, 1] < FH)
        td_paths[lbl] = td[valid].tolist()

    labels = [l for l in td_paths if td_paths[l]]

    # ── 1. Heatmap cenital ─────────────────────────────────────────────────
    field_bg = _draw_field_2d(FW, FH)
    heatmap_canvas = field_bg.copy().astype(np.float32)

    ind_hmaps = {}
    for lbl in labels:
        pts = td_paths[lbl]
        hmap = _gaussian_heatmap(pts, FH, FW, sigma=args.sigma)
        ind_hmaps[lbl] = hmap

    # Superposición con colores por objeto
    for lbl, hmap in ind_hmaps.items():
        color = COLORS.get(lbl, (200, 200, 200))
        layer = np.zeros((FH, FW, 3), dtype=np.float32)
        layer[:] = color[::-1]   # BGR
        mask = hmap[:, :, None]
        heatmap_canvas = (heatmap_canvas * (1 - mask * 0.65) +
                          layer             * mask * 0.65)
    heatmap_canvas = heatmap_canvas.clip(0, 255).astype(np.uint8)

    # ── 2. Trails cenitales ────────────────────────────────────────────────
    trails_canvas = field_bg.copy()
    for lbl in labels:
        pts = td_paths[lbl]
        if len(pts) < 2:
            continue
        color_end = COLORS.get(lbl, (200, 200, 200))
        n = len(pts)
        for i in range(1, n):
            t = i / n
            r = int(80 + t * (color_end[0] - 80))
            g = int(80 + t * (color_end[1] - 80))
            b = int(80 + t * (color_end[2] - 80))
            cv2.line(trails_canvas,
                     (int(pts[i-1][0]), int(pts[i-1][1])),
                     (int(pts[i][0]),   int(pts[i][1])),
                     (b, g, r), 2 if lbl == "ball" else 3, cv2.LINE_AA)
        # Inicio (gris) y fin (color)
        cv2.circle(trails_canvas, (int(pts[0][0]), int(pts[0][1])), 6, (160,160,160), -1)
        cv2.circle(trails_canvas, (int(pts[-1][0]), int(pts[-1][1])), 7,
                   color_end[::-1], -1)

    # ── 3. Voronoi cenital ─────────────────────────────────────────────────
    voronoi_canvas = field_bg.copy()
    robot_td_pts = {lbl: td_paths[lbl] for lbl in labels if lbl.startswith("robot")}

    if robot_td_pts:
        # Última posición conocida de cada robot
        last_pts = {lbl: pts[-1] for lbl, pts in robot_td_pts.items() if pts}
        if len(last_pts) >= 2:
            subdiv = cv2.Subdiv2D((0, 0, FW, FH))
            lbl_list = sorted(last_pts.keys())
            for lbl in lbl_list:
                px, py = float(last_pts[lbl][0]), float(last_pts[lbl][1])
                subdiv.insert((px, py))
            facets, centers = subdiv.getVoronoiFacetList([])
            for i, facet in enumerate(facets):
                lbl = lbl_list[i % len(lbl_list)]
                color = COLORS.get(lbl, (128, 128, 128))
                poly  = np.array(facet, dtype=np.int32)
                overlay = voronoi_canvas.copy()
                cv2.fillPoly(overlay, [poly], color[::-1])
                cv2.addWeighted(overlay, 0.35, voronoi_canvas, 0.65, 0, voronoi_canvas)
                cv2.polylines(voronoi_canvas, [poly], True, (255,255,255), 1, cv2.LINE_AA)
        # Dibujar posiciones finales
        for lbl, pos in last_pts.items():
            color = COLORS.get(lbl, (200,200,200))
            cv2.circle(voronoi_canvas, (int(pos[0]), int(pos[1])), 10, color[::-1], -1)
            cv2.circle(voronoi_canvas, (int(pos[0]), int(pos[1])), 12, (255,255,255), 2)
            cv2.putText(voronoi_canvas, lbl, (int(pos[0])+12, int(pos[1])-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[::-1], 2, cv2.LINE_AA)
        if "ball" in td_paths and td_paths["ball"]:
            bp = td_paths["ball"][-1]
            cv2.circle(voronoi_canvas, (int(bp[0]), int(bp[1])), 7,
                       COLORS["ball"][::-1], -1)

    # ── 4. Panel unificado ─────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        # 2×2: heatmap | voronoi
        #       trails | stats
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))

        def _show(ax, img, title):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
            ax.axis("off")

        _show(axes[0, 0], heatmap_canvas, "Heatmap de actividad (cenital)")
        _show(axes[0, 1], voronoi_canvas, "Zonas de control — Voronoi")
        _show(axes[1, 0], trails_canvas,  "Trayectorias (cenital)")

        # Panel de estadísticas
        ax_s = axes[1, 1]
        ax_s.set_facecolor("#1a1a2e")
        ax_s.axis("off")
        ax_s.set_title("Estadísticas", fontsize=12, fontweight="bold",
                        color="white", pad=6)

        robot_labels_sorted = sorted(
            [l for l in summary["possession"] if l.startswith("robot")]
        )
        y = 0.92
        for lbl in robot_labels_sorted:
            pdata = summary["possession"][lbl]
            spd   = summary["speed_avg_px_s"].get(lbl, 0)
            dist_ = summary["distance_px"].get(lbl, 0)
            rgb   = [x/255 for x in COLORS.get(lbl, (200,200,200))]
            ax_s.text(0.05, y, f"■ {lbl}", transform=ax_s.transAxes,
                      fontsize=11, color=rgb, fontweight="bold")
            y -= 0.07
            for line in [
                f"  Posesión:  {pdata['pct']:.1f}%  ({pdata['frames']} frames)",
                f"  Velocidad: {spd} px/s",
                f"  Distancia: {dist_:.0f} px",
            ]:
                ax_s.text(0.05, y, line, transform=ax_s.transAxes,
                          fontsize=9.5, color="white")
                y -= 0.055
            y -= 0.02

        # Eventos
        ax_s.text(0.05, y, "Eventos:", transform=ax_s.transAxes,
                  fontsize=11, color="#ffd700", fontweight="bold")
        y -= 0.065
        for ev in data["events"]:
            if ev["type"] == "pass":
                line = f"  Pase {ev['time_s']}s: {ev['from']}→{ev['to']}"
            elif ev["type"] == "collision":
                line = f"  Colisión {ev['time_s']}s: {'+'.join(ev['robots'])}"
            else:
                line = f"  {ev['type']} {ev['time_s']}s"
            ax_s.text(0.05, y, line, transform=ax_s.transAxes,
                      fontsize=9, color="#cccccc")
            y -= 0.055
            if y < 0.05:
                break

        fig.suptitle("Análisis cenital del partido — Copa FutBotMX",
                     fontsize=14, fontweight="bold", color="white",
                     y=0.99)
        fig.patch.set_facecolor("#12121e")
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        panel_path = os.path.join(args.output, "topdown_panel.png")
        fig.savefig(panel_path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  → {panel_path}")

    except ImportError:
        print("  (matplotlib no disponible — guardando imágenes individuales)")
        cv2.imwrite(os.path.join(args.output, "topdown_heatmap.jpg"), heatmap_canvas)
        cv2.imwrite(os.path.join(args.output, "topdown_trails.jpg"),  trails_canvas)
        cv2.imwrite(os.path.join(args.output, "topdown_voronoi.jpg"), voronoi_canvas)

    # Guardar imágenes individuales siempre
    cv2.imwrite(os.path.join(args.output, "topdown_heatmap.jpg"), heatmap_canvas)
    cv2.imwrite(os.path.join(args.output, "topdown_trails.jpg"),  trails_canvas)
    cv2.imwrite(os.path.join(args.output, "topdown_voronoi.jpg"), voronoi_canvas)
    print(f"  → {args.output}/topdown_heatmap.jpg")
    print(f"  → {args.output}/topdown_trails.jpg")
    print(f"  → {args.output}/topdown_voronoi.jpg")


def cmd_trails(args):
    """Trayectorias con degradado de color azul→rojo (pasado→presente)."""
    with open(args.analytics) as f:
        data = json.load(f)

    paths = data["paths"]
    os.makedirs(args.output, exist_ok=True)

    bg   = _load_bg(args.bg)
    H, W = bg.shape[:2]
    vis  = bg.copy()

    for lbl, pts in paths.items():
        if len(pts) < 2:
            continue
        color_end   = COLORS.get(lbl, (200, 200, 200))
        n           = len(pts)
        for i in range(1, n):
            t   = i / n                  # 0 (inicio) → 1 (fin)
            # Degradado desde gris oscuro hasta el color del objeto
            r   = int(80 + t * (color_end[0] - 80))
            g   = int(80 + t * (color_end[1] - 80))
            b   = int(80 + t * (color_end[2] - 80))
            thickness = 2 if lbl == "ball" else 3
            cv2.line(vis,
                     (int(pts[i-1][0]), int(pts[i-1][1])),
                     (int(pts[i][0]),   int(pts[i][1])),
                     (b, g, r), thickness, cv2.LINE_AA)
        # Marcador de inicio y fin
        cv2.circle(vis, (int(pts[0][0]),  int(pts[0][1])),  8, (200, 200, 200), -1)
        cv2.circle(vis, (int(pts[-1][0]), int(pts[-1][1])), 8, color_end[::-1], -1)
        cv2.putText(vis, lbl, (int(pts[-1][0]) + 10, int(pts[-1][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_end[::-1], 2, cv2.LINE_AA)

    out_path = os.path.join(args.output, "trails.jpg")
    cv2.imwrite(out_path, vis)
    print(f"  → {out_path}")


def cmd_voronoi(args):
    """
    Diagrama de Voronoi dinámico: para cada frame renderiza las zonas de
    control de cada robot y superpone la posición de todos los objetos.
    Genera un video MP4 de resultado.
    """
    with open(args.analytics) as f:
        data = json.load(f)

    frames_data  = data["frames"]
    frames_dir   = args.frames_dir
    os.makedirs(args.output, exist_ok=True)

    fidxs = sorted(frames_data.keys(), key=int)
    if not fidxs:
        print("Sin frames en analytics"); return

    # Determinar tamaño del frame
    sample_path = os.path.join(frames_dir, f"{int(fidxs[0]):05d}.jpg")
    bg_sample   = cv2.imread(sample_path)
    if bg_sample is None:
        print(f"No se pudo leer frame de ejemplo: {sample_path}"); return
    H, W = bg_sample.shape[:2]

    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(args.output, "voronoi.mp4")
    writer     = cv2.VideoWriter(video_path, fourcc, args.fps, (W, H))

    for fidx_str in fidxs:
        fidx  = int(fidx_str)
        fdata = frames_data[fidx_str]

        frame_path = os.path.join(frames_dir, f"{fidx:05d}.jpg")
        bg = cv2.imread(frame_path)
        if bg is None:
            continue
        vis = bg.copy()

        # Recoger posiciones de robots
        robot_pts   = []
        robot_colors = []
        for lbl, color in COLORS.items():
            if not lbl.startswith("robot"):
                continue
            # Buscar centroide en el frame de analytics
            # (no está directamente, usamos paths pero por frame no está disponible aquí;
            #  reconstruimos desde frames_data)

        # Buscar posiciones en el frame tracks original (necesitamos el tracks JSON)
        # Como alternativa robusta, usar ball_pos y el campo de events
        # Para Voronoi simplificado, usar pos del frame del analytics (ball_pos + ball_dists)
        # Si no tenemos las pos por robot en analytics frames, saltar Voronoi per-frame
        # y generar solo el frame final con los paths.
        # Por ahora generamos el overlay básico con el ball_pos
        ball_pos = fdata.get("ball_pos")
        poss     = fdata.get("possessor")

        if ball_pos:
            cv2.circle(vis, (int(ball_pos[0]), int(ball_pos[1])), 10,
                       COLORS["ball"][::-1], -1)

        if poss:
            color = COLORS.get(poss, (200,200,200))
            cv2.putText(vis, f"Posesion: {poss}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color[::-1], 3, cv2.LINE_AA)

        writer.write(vis)

    writer.release()
    print(f"  → {video_path}")


# ── Video side-by-side ────────────────────────────────────────────────────────

TRAIL_LEN    = 30     # posiciones pasadas a mostrar en la estela
PANEL_W      = 960    # ancho de cada panel (salida total: 1920)
PANEL_H      = 540    # alto (salida: 960×540 compatible con full-HD partida)
FLASH_FRAMES = 18     # frames de destello en un gol


def cmd_video(args):
    """
    Genera un video MP4 con dos paneles en paralelo:
      Izquierda — frame original con bounding boxes, centroides, marcador, tiempo.
      Derecha   — campo sintético cenital con trails animados y Voronoi simplificado.
    No requiere homografía: mapea coordenadas de cámara directamente al canvas.
    Si se pasa --corners usa homografía para vista cenital precisa.
    """
    with open(args.tracks) as f:
        tracks = json.load(f)
    with open(args.analytics) as f:
        data = json.load(f)

    af_frames   = data["frames"]
    events_list = data["events"]

    fidxs = sorted(tracks.keys(), key=int)

    # Dimensiones del frame fuente
    first_path = os.path.join(args.frames_dir, f"{int(fidxs[0]):05d}.jpg")
    sample = cv2.imread(first_path)
    if sample is None:
        print(f"No se puede leer frame de muestra: {first_path}"); return
    src_H, src_W = sample.shape[:2]

    # Homografía opcional
    H_mat = None
    if args.corners and os.path.exists(args.corners):
        H_mat, _, _ = _build_homography(args.corners, PANEL_W, PANEL_H)
        print(f"  Homografia cargada: {args.corners}")

    def to_td(cx, cy):
        if H_mat is not None:
            pt  = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
            out = cv2.perspectiveTransform(pt, H_mat)
            return (int(out[0, 0, 0]), int(out[0, 0, 1]))
        return (int(cx * PANEL_W / src_W), int(cy * PANEL_H / src_H))

    # Eventos indexados por frame
    events_by_frame = {}
    for ev in events_list:
        events_by_frame.setdefault(ev["frame"], []).append(ev)

    os.makedirs(args.output, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(args.tracks))[0].replace("_tracks", "")
    video_path = os.path.join(args.output, f"{video_name}_sidebyside.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, args.fps, (PANEL_W * 2, PANEL_H))
    print(f"  Generando {video_path}  ({len(fidxs)} frames @ {args.fps} fps)…")

    goal_scores  = {}
    ball_trail   = []          # (tx, ty) en espacio cenital
    robot_trails = {}          # label → [(tx,ty),…]
    flash_queue  = []          # [[event, frames_restantes],…]
    dt           = args.step / args.fps

    for fidx_str in fidxs:
        fidx = int(fidx_str)
        tf   = tracks[fidx_str]
        af   = af_frames.get(str(fidx), {})
        poss = af.get("possessor")

        # Actualizar marcador y cola de destellos
        for ev in events_by_frame.get(fidx, []):
            if ev["type"] == "goal":
                goal_scores = dict(ev.get("score", goal_scores))
                flash_queue.append([ev, FLASH_FRAMES])
            elif ev["type"] == "shot_on_goal":
                flash_queue.append([ev, FLASH_FRAMES // 3])
        flash_queue = [[ev, r - 1] for ev, r in flash_queue if r > 1]

        # ── Panel izquierdo: frame original con overlays ──────────────────
        frame_path = os.path.join(args.frames_dir, f"{fidx:05d}.jpg")
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        left = frame.copy()
        H_f, W_f = left.shape[:2]

        for lbl, obj in tf.items():
            color = COLORS.get(lbl, (200, 200, 200))
            box   = obj.get("box_xyxy")
            ctr   = obj.get("centroid")
            if box:
                x1, y1, x2, y2 = [int(v) for v in box]
                thick = 3 if lbl == poss else 2
                cv2.rectangle(left, (x1, y1), (x2, y2), color[::-1], thick)
                cv2.putText(left, lbl, (x1, max(y1 - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color[::-1], 2, cv2.LINE_AA)
            if ctr:
                r = 6 if lbl == "ball" else 9
                cv2.circle(left, (int(ctr[0]), int(ctr[1])), r, color[::-1], -1)

        # Marcador (esquina superior izquierda)
        if goal_scores:
            score_txt = "  ".join(f"{lbl}: {g}" for lbl, g in goal_scores.items())
            (tw, _), _ = cv2.getTextSize(score_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(left, (8, 8), (16 + tw, 52), (0, 0, 0), -1)
            cv2.putText(left, score_txt, (14, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Tiempo (esquina superior derecha)
        time_s = round(fidx * dt, 1)
        cv2.putText(left, f"{time_s:.1f}s", (W_f - 130, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2, cv2.LINE_AA)

        # Barra de posesión (parte inferior)
        if poss:
            pc = COLORS.get(poss, (200, 200, 200))
            cv2.rectangle(left, (0, H_f - 10), (W_f, H_f), pc[::-1], -1)
            cv2.putText(left, f"Posesion: {poss}", (20, H_f - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, pc[::-1], 2, cv2.LINE_AA)

        # Destello de gol
        for ev, remaining in flash_queue:
            if ev["type"] == "goal":
                alpha = min(remaining / FLASH_FRAMES, 1.0) * 0.45
                ov = left.copy()
                cv2.rectangle(ov, (0, 0), (W_f, H_f), (0, 30, 220), -1)
                cv2.addWeighted(ov, alpha, left, 1 - alpha, 0, left)
                cv2.putText(left, "GOL!", (W_f // 2 - 140, H_f // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.5, (255, 255, 255), 10, cv2.LINE_AA)

        left = cv2.resize(left, (PANEL_W, PANEL_H))

        # ── Panel derecho: campo cenital sintético ────────────────────────
        td = _draw_field_2d(PANEL_W, PANEL_H)

        # Actualizar y dibujar trails
        for lbl, obj in tf.items():
            ctr = obj.get("centroid")
            if not ctr:
                continue
            tx, ty = to_td(ctr[0], ctr[1])
            if lbl == "ball":
                ball_trail.append((tx, ty))
                if len(ball_trail) > TRAIL_LEN:
                    ball_trail[:] = ball_trail[-TRAIL_LEN:]
            elif lbl.startswith("robot"):
                robot_trails.setdefault(lbl, []).append((tx, ty))
                if len(robot_trails[lbl]) > TRAIL_LEN:
                    robot_trails[lbl][:] = robot_trails[lbl][-TRAIL_LEN:]

        for lbl, trail in robot_trails.items():
            if len(trail) < 2:
                continue
            color = COLORS.get(lbl, (200, 200, 200))
            n = len(trail)
            for i in range(1, n):
                a = i / n
                c = tuple(int(v * a * 0.85) for v in color[::-1])
                cv2.line(td, trail[i - 1], trail[i], c, 2, cv2.LINE_AA)

        if len(ball_trail) >= 2:
            n = len(ball_trail)
            for i in range(1, n):
                a = i / n
                c = tuple(int(v * a * 0.85) for v in COLORS["ball"][::-1])
                cv2.line(td, ball_trail[i - 1], ball_trail[i], c, 2, cv2.LINE_AA)

        # Robots
        for lbl, obj in tf.items():
            if not lbl.startswith("robot"):
                continue
            ctr = obj.get("centroid")
            if not ctr:
                continue
            tx, ty = to_td(ctr[0], ctr[1])
            color  = COLORS.get(lbl, (200, 200, 200))
            if lbl == poss:
                cv2.circle(td, (tx, ty), 20, (0, 220, 220), 3)
            cv2.circle(td, (tx, ty), 13, color[::-1], -1)
            cv2.circle(td, (tx, ty), 14, (255, 255, 255), 1)
            cv2.putText(td, lbl[-1], (tx - 5, ty + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Balón
        for lbl, obj in tf.items():
            if lbl != "ball":
                continue
            ctr = obj.get("centroid")
            if not ctr:
                continue
            tx, ty = to_td(ctr[0], ctr[1])
            cv2.circle(td, (tx, ty), 7, COLORS["ball"][::-1], -1)
            cv2.circle(td, (tx, ty), 8, (255, 255, 255), 1)

        # Marcador en panel cenital
        y_txt = 18
        for lbl, g in goal_scores.items():
            color = COLORS.get(lbl, (200, 200, 200))
            cv2.putText(td, f"{lbl}: {g}", (8, y_txt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[::-1], 2, cv2.LINE_AA)
            y_txt += 20

        # Destello en panel cenital
        for ev, remaining in flash_queue:
            if ev["type"] == "goal":
                alpha = min(remaining / FLASH_FRAMES, 1.0) * 0.35
                ov = td.copy()
                cv2.rectangle(ov, (0, 0), (PANEL_W, PANEL_H), (0, 0, 200), -1)
                cv2.addWeighted(ov, alpha, td, 1 - alpha, 0, td)

        # ── Unir paneles ──────────────────────────────────────────────────
        combined = np.hstack([left, td])
        cv2.line(combined, (PANEL_W, 0), (PANEL_W, PANEL_H), (180, 180, 180), 2)
        writer.write(combined)

        if fidx % 20 == 0:
            print(f"    frame {fidx:04d}/{fidxs[-1]}  t={time_s:.1f}s")

    writer.release()
    print(f"  -> {video_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd")

    # heatmap
    p_h = sub.add_parser("heatmap", help="Heatmaps de densidad sobre frame de cámara")
    p_h.add_argument("--analytics", required=True)
    p_h.add_argument("--bg",        default="", help="Imagen de fondo (frame del video)")
    p_h.add_argument("--output",    default="output/viz/")
    p_h.add_argument("--sigma",     type=int, default=40, help="Suavizado gaussiano (px)")

    # topdown
    p_td = sub.add_parser("topdown", help="Vista cenital 2D con heatmap + trails + Voronoi + stats")
    p_td.add_argument("--analytics", required=True)
    p_td.add_argument("--corners",   required=True, help="JSON con las 4 esquinas del campo")
    p_td.add_argument("--output",    default="output/viz/")
    p_td.add_argument("--sigma",     type=int, default=30, help="Suavizado gaussiano cenital (px)")

    # trails
    p_t = sub.add_parser("trails", help="Trayectorias con degradado temporal")
    p_t.add_argument("--analytics", required=True)
    p_t.add_argument("--bg",        default="")
    p_t.add_argument("--output",    default="output/viz/")

    # voronoi
    p_v = sub.add_parser("voronoi", help="Zonas de control (Voronoi) por frame")
    p_v.add_argument("--analytics",  required=True)
    p_v.add_argument("--frames_dir", required=True)
    p_v.add_argument("--output",     default="output/viz/")
    p_v.add_argument("--fps",        type=float, default=10.0)

    # video (side-by-side)
    p_vid = sub.add_parser("video", help="Video side-by-side: original | vista cenital")
    p_vid.add_argument("--tracks",     required=True, help="JSON de tracks (pipeline.py)")
    p_vid.add_argument("--analytics",  required=True, help="JSON de analytics")
    p_vid.add_argument("--frames_dir", required=True, help="Directorio de frames")
    p_vid.add_argument("--corners",    default="",    help="JSON de esquinas del campo (opcional)")
    p_vid.add_argument("--output",     default="output/videos/")
    p_vid.add_argument("--fps",        type=float, default=10.0,
                       help="FPS del video de salida (10 fps es fluido con step=3)")
    p_vid.add_argument("--step",       type=int,   default=3,
                       help="Paso de extraccion usado (para calcular tiempo real)")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help(); sys.exit(1)

    if args.cmd == "heatmap":
        cmd_heatmap(args)
    elif args.cmd == "topdown":
        cmd_topdown(args)
    elif args.cmd == "trails":
        cmd_trails(args)
    elif args.cmd == "voronoi":
        cmd_voronoi(args)
    elif args.cmd == "video":
        cmd_video(args)


if __name__ == "__main__":
    main()
