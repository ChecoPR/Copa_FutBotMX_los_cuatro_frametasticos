# Copa FutBotMX — Los Cuatro Frametásticos

## Tabla de contenidos

1. [Resumen del sistema](#1-resumen-del-sistema)
2. [Instalación y configuración](#2-instalación-y-configuración)
3. [Pipeline paso a paso](#3-pipeline-paso-a-paso)
4. [Descripción de scripts](#4-descripción-de-scripts)
5. [Fórmulas y algoritmos](#5-fórmulas-y-algoritmos)
6. [Estructura de salidas](#6-estructura-de-salidas)
7. [Resultados sobre IMG_9866](#7-resultados-sobre-img_9866)
8. [Innovaciones técnicas](#8-innovaciones-técnicas)
9. [Reproducibilidad](#9-reproducibilidad)

---

## 1. Resumen del sistema

Pipeline de visión por computadora para analizar partidos de fútbol robótico grabados en video `.MOV`. Detecta robots y balón, construye trayectorias, califica equipos por color, detecta eventos de juego (goles, tiros, pases, colisiones) y genera visualizaciones interactivas.

```
Video .MOV
    │
    ▼ extract_frames.py
Frames JPG (step=3)
    │
    ▼ pipeline.py --auto
Tracks JSON + Máscaras SAM3 + Video con overlays
    │
    ▼ auto_corners.py
Homografía perspectiva → vista cenital
    │
    ▼ analytics.py
Analytics JSON (posesión, velocidad, eventos, goles)
    │
    ▼ visualize.py
Heatmaps · Voronoi · Trayectorias · Video side-by-side
```

### Arquitectura de detección y tracking

| Objeto | Detector | Tracker |
|--------|----------|---------|
| Robots | YOLOv8 fine-tuned (`runs/detect/train-2/weights/best.pt`) | Proximidad + zona de fusión + Hungarian assignment |
| Balón | SAM3 (Meta AI) VG por bounding box | Propagación automática SAM3 |
| Porterías | HSV (azul/amarillo) + mediana temporal | Detección estática por color |

### Panel de análisis completo

![Panel cenital — heatmap, Voronoi, trayectorias y estadísticas](output/viz/IMG_9866/topdown_panel.png)

*Vista cenital del partido: heatmap de actividad, zonas de control Voronoi, trayectorias de los robots y estadísticas del juego.*

---

## 2. Instalación y configuración

### Requisitos de hardware

| GPU | VRAM | Compute | Uso |
|-----|------|---------|-----|
| GTX 1080 (GPU 0) | 8 GB | sm_61 | Disponible |
| TITAN X (GPU 1) | 11 GB | sm_52 | **En uso** (SAM3) |

> **WSL2**: NCCL multi-GPU falla → siempre `gpus_to_use=[1]`

### Entorno Python

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Parches aplicados a SAM3 (GPU TITAN X sm_52)

Los siguientes archivos tienen `bfloat16 → float16` y `Triton → scipy`:

| Archivo | Línea | Cambio |
|---------|-------|--------|
| `sam3/sam3/model/sam3_base_predictor.py` | 205 | `float16` |
| `sam3/sam3/model/sam3_video_inference.py` | 800, 908 | `float16` |
| `sam3/sam3/model/sam3_tracking_predictor.py` | 50, 1097, 1148 | `float16` |
| `sam3/sam3/model/sam3_image.py` | 864 | `float16` |
| `sam3/sam3/perflib/connected_components.py` | — | fallback scipy CPU |

---

## 3. Pipeline paso a paso

### Etapa 1 — Extracción de frames

```bash
python scripts/extract_frames.py \
    --video /mnt/d/videos/IMG_9866.MOV \
    --step 3
# Salida: output/frames/IMG_9866/00000.jpg … 00101.jpg
```

`--step 3` extrae 1 de cada 3 frames → a 30 fps original = 10 fps efectivos.

### Etapa 2 — Segmentación y tracking (pipeline)

```bash
python scripts/pipeline.py \
    --frames_dir output/frames/IMG_9866 \
    --auto
# Salida:
#   output/tracks/IMG_9866_tracks.json
#   output/masks/IMG_9866/
#   output/videos/IMG_9866_tracked.mp4
```

El modo `--auto` detecta objetos automáticamente con YOLO en el primer frame.

**Detección automática — frame 0:**

![YOLO detecta robot1, robot2 y balón en el primer frame](output/debug/IMG_9866_autodetect.jpg)

*YOLOv8 fine-tuned identifica robot1 (azul), robot2 (naranja) y el balón (amarillo) con sus centroides en píxeles.*

**Tracking en frame 3 (partido en curso):**

![Tracking activo: robot1, robot2 y balón con bounding boxes](output/debug/IMG_9866_tracked_frame3.jpg)

*SAM3 propaga la máscara del balón frame a frame; YOLO re-detecta robots con Hungarian assignment para evitar intercambio de identidades.*

### Etapa 3 — Detección de esquinas del campo (homografía)

```bash
python scripts/auto_corners.py \
    --frame  output/frames/IMG_9866/00000.jpg \
    --out    output/field_corners_IMG_9866.json \
    --debug  output/debug/IMG_9866_corners_debug.jpg
```

Si la detección automática falla (cámara en ángulo severo), edita el JSON manualmente:

```json
{
  "frame_size": [1920, 1080],
  "corners_image": {
    "top_left":     [90,   0],
    "top_right":    [1775, 5],
    "bottom_right": [1875, 575],
    "bottom_left":  [290,  470]
  }
}
```

![Detección de esquinas y porterías por color](output/debug/debug_corners_IMG_9866.jpg)

*Las líneas verdes definen el cuadrilátero del campo (homografía). Los puntos YELLOW y BLUE son los centroides de las porterías detectadas por mediana temporal HSV. La miniatura muestra la vista cenital resultante.*

### Etapa 4 — Analytics

```bash
python scripts/analytics.py \
    --tracks     output/tracks/IMG_9866_tracks.json \
    --frames_dir output/frames/IMG_9866 \
    --fps 30 --step 3
# Salida: output/analytics/IMG_9866_analytics.json
```

### Etapa 5 — Visualizaciones

```bash
# Video side-by-side: original | vista cenital animada
python scripts/visualize.py video \
    --tracks     output/tracks/IMG_9866_tracks.json \
    --analytics  output/analytics/IMG_9866_analytics.json \
    --frames_dir output/frames/IMG_9866 \
    --corners    output/field_corners_IMG_9866.json \
    --output     output/videos/ \
    --fps 10 --step 3

# Heatmaps de actividad
python scripts/visualize.py heatmap \
    --analytics output/analytics/IMG_9866_analytics.json \
    --bg        output/frames/IMG_9866/00000.jpg \
    --output    output/viz/IMG_9866/

# Panel cenital: heatmap + Voronoi + trails + estadísticas
python scripts/visualize.py topdown \
    --analytics output/analytics/IMG_9866_analytics.json \
    --corners   output/field_corners_IMG_9866.json \
    --output    output/viz/IMG_9866/
```

---

## 4. Descripción de scripts

| Script | Función |
|--------|---------|
| `extract_frames.py` | Extrae frames de video `.MOV` a JPEG con paso configurable |
| `pipeline.py` | Pipeline principal: YOLO (robots) + SAM3 (balón) → tracks + máscaras + video |
| `auto_corners.py` | Detecta las 4 esquinas del campo por líneas blancas HSV + posición de porterías |
| `analytics.py` | Calcula posesión, velocidad, eventos (pase, colisión, gol, tiro) |
| `visualize.py` | Heatmaps, Voronoi, trayectorias, video side-by-side con vista cenital |
| `auto_detect.py` | Detección YOLO del primer frame para inicialización automática |
| `pick_points.py` | Herramienta interactiva para seleccionar prompts de punto (requiere GUI) |
| `pick_field_corners.py` | Herramienta interactiva para marcar esquinas (requiere GUI) |
| `yolo_sam3_tracker.py` | Prototipo de integración YOLO → SAM3 (ensamble) |

---

## 5. Fórmulas y algoritmos

### 5.1 Tracking de robots — Zona de fusión

Cuando los robots están separados (`d > MERGE_DIST = 160 px`):

```
velocidad_i(t) = regresión lineal OLS sobre historial de posiciones (12 frames)
prediccion_i(t) = posicion_i(t-1) + velocidad_i(t)
asignacion = argmin Hungarian( cost_matrix[i,j] = ||prediccion_i - deteccion_j|| )
```

Cuando los robots se juntan (`d < 160 px`, **zona de fusión**):

```
pre_snap = snapshot(posicion, velocidad) al entrar a la zona
prediccion_i(t) = pre_snap_i.posicion + pre_snap_i.velocidad × frames_en_zona
```

Este mecanismo preserva las identidades aunque los robots estén superpuestos.

### 5.2 Posesión

```
posesion(t) = argmin_robot { ||centroide_robot - centroide_balon|| }
    si esa distancia < POSSESSION_DIST_PX = 150 px
```

### 5.3 Velocidad y distancia

```
velocidad_i(t) [px/s] = ||centroide_i(t) - centroide_i(t-1)|| × fps_efectivos

fps_efectivos = fps_original / step = 30 / 3 = 10 fps

distancia_i = Σ_t ||centroide_i(t) - centroide_i(t-1)||
```

### 5.4 Detección de eventos

**Pase** — cambio de posesor con cooldown de 3 frames:
```
pase(t): posesion(t) ≠ posesion(t-1) AND posesion(t) ≠ None
```

**Colisión** — solape de bounding boxes:
```
IoU(box_robot1, box_robot2) > COLLISION_IOU_THRESH = 0.05
```

**Gol** — balón dentro de la máscara de la portería (o dentro de su bounding box
con margen de 80 px), con cooldown de 30 frames:
```
gol(t): centroide_balon ∈ mascara_porteria(color)
    OR  centroide_balon ∈ bbox_porteria(color) ± GOAL_BBOX_MARGIN
```

**Tiro a gol** — predicción de trayectoria antes de que el balón entre:
```
puntos = historial_balon[-SHOT_HISTORY:]   # últimos 6 frames
vx, vy = polyfit(t, x, 1)[0], polyfit(t, y, 1)[0]   # vel. por regresión lineal
speed = √(vx² + vy²) > SHOT_MIN_SPEED = 8 px/frame

Para k = 1..SHOT_LOOKAHEAD (18 frames):
    px(k) = pos_balon + vx·k
    py(k) = pos_balon + vy·k
    si (px, py) ∈ bbox_porteria ± SHOT_MARGIN → TIRO detectado
```

### 5.5 Detección de porterías — Mediana temporal HSV

```
Para N_MEDIAN = 20 frames muestreados uniformemente:
    mascara_color(frame) = HSV_inRange(frame, LO, HI)

mascara_estable = pixel_mediana_temporal(mascaras)
    # Solo píxeles estacionarios sobreviven la mediana → elimina robots/personas en movimiento
```

Rangos HSV:

| Color | H | S | V |
|-------|---|---|---|
| Amarillo | 15–38 | >100 | >100 |
| Azul | 95–130 | >100 | >40 |

### 5.6 Homografía para vista cenital

Se calculan 4 correspondencias campo→canvas:

```
src (píxeles cámara):  [TL, TR, BR, BL]
dst (canvas 800×540):  [(0,0), (800,0), (800,540), (0,540)]

H = findHomography(src, dst)

punto_canvas = perspectiveTransform(punto_camara, H)
```

Para el video IMG_9866 (cámara a 45°, borde superior fuera de frame):
```
TL = (90, 0)    TR = (1775, 5)
BR = (1875, 575) BL = (290, 470)
```

---

## 6. Estructura de salidas

```
output/
├── frames/
│   └── <video>/          # Frames extraídos (JPEG, no versionar en git)
│       ├── 00000.jpg
│       └── …
├── masks/
│   └── <video>/          # Máscaras binarias SAM3 por objeto
│       ├── 00000_obj3.png  # obj3 = balón
│       └── …
├── tracks/
│   └── <video>_tracks.json   # Centroides + boxes por frame
├── analytics/
│   └── <video>_analytics.json  # Posesión, velocidad, eventos
├── videos/
│   ├── <video>_tracked.mp4      # Video con overlays SAM3/YOLO
│   └── <video>_sidebyside.mp4   # Video side-by-side con vista cenital
├── viz/
│   └── <video>/
│       ├── heatmap_robot1.jpg
│       ├── heatmap_robot2.jpg
│       ├── heatmap_ball.jpg
│       ├── heatmap_combined.jpg
│       ├── heatmap_panel.png
│       ├── topdown_heatmap.jpg
│       ├── topdown_trails.jpg
│       ├── topdown_voronoi.jpg
│       └── topdown_panel.png
├── debug/
│   ├── <video>_autodetect.jpg   # Detecciones YOLO frame 0
│   └── <video>_corners_debug.jpg
└── field_corners_<video>.json   # Esquinas del campo (homografía)
```

### Formato de `_tracks.json`

```json
{
  "0": {
    "robot1": {
      "label": "robot1",
      "score": 0.93,
      "source": "yolo",
      "centroid": [849, 295],
      "box_xyxy": [789, 199, 909, 392]
    },
    "ball": {
      "label": "ball",
      "score": 0.95,
      "source": "sam3",
      "centroid": [1084, 267],
      "box_xyxy": null
    }
  }
}
```

### Formato de `_analytics.json`

```json
{
  "frames": {
    "0": { "possessor": "robot2", "ball_pos": [1084,267], "velocities": {…}, "events": [] }
  },
  "summary": {
    "total_frames": 102,
    "possession": { "robot1": {"frames":72,"pct":70.6}, … },
    "speed_avg_px_s": { "robot1": 102.8, "robot2": 65.5, "ball": 25.9 },
    "distance_px":    { "robot1": 1048.7, "robot2": 667.8, "ball": 263.9 },
    "score": { "robot1": 2, "robot2": 0 }
  },
  "events": [ … ],
  "paths": { "robot1": [[cx,cy],…], "robot2": […], "ball": […] }
}
```

---

## 7. Resultados sobre IMG_9866

Video de prueba: `IMG_9866.MOV` (102 frames extraídos, step=3 → 10 fps efectivos ≈ 10.2 s)

### Métricas cuantitativas

| Métrica | Robot 1 | Robot 2 | Balón |
|---------|---------|---------|-------|
| Posesión | **70.6%** (72 frames) | 2.0% (2 frames) | — |
| Velocidad prom | **102.8 px/s** | 65.5 px/s | 25.9 px/s |
| Distancia total | **1048.7 px** | 667.8 px | 263.9 px |
| Goles marcados | **2** | 0 | — |

### Eventos detectados

| Tiempo | Evento | Detalle |
|--------|--------|---------|
| 0.6 s | Pase | robot2 → robot1 |
| 1.0 s | Colisión | robot1 + robot2 (IoU = 0.08) |
| 1.6 s | Tiro a gol | portería amarilla/derecha (~1 frame antes) |
| 1.7 s | **GOL** | robot1 · portería derecha (amarilla) · 1–0 |
| 4.7 s | Tiro a gol | portería amarilla/derecha (~1 frame antes) |
| 5.1 s | **GOL** | robot1 · portería derecha (amarilla) · 2–0 |

### Análisis cenital completo

![Panel cenital: heatmap de actividad, zonas Voronoi, trayectorias y estadísticas](output/viz/IMG_9866/topdown_panel.png)

### Heatmaps de actividad (vista cámara)

![Heatmaps de actividad por objeto sobre el frame real](output/viz/IMG_9866/heatmap_panel.png)

*Arriba: robot1 (70.6% posesión, 102.8 px/s) y robot2 (2.0% posesión, 65.5 px/s). Abajo: balón y mapa combinado. El color cálido indica mayor tiempo de permanencia.*

### Trayectorias (vista cenital)

| Trayectorias | Zonas de control |
|:---:|:---:|
| ![Trayectorias de robots y balón](output/viz/IMG_9866/topdown_trails.jpg) | ![Diagrama de Voronoi](output/viz/IMG_9866/topdown_voronoi.jpg) |
| *Trails: robot1 (naranja), robot2 (azul), balón (gris). Ambos robots atacan hacia la portería derecha.* | *Voronoi: robot1 controla el lado derecho (oliva), robot2 el lado izquierdo (teal).* |

### Archivos generados

| Archivo | Descripción |
|---------|-------------|
| `output/tracks/IMG_9866_tracks.json` | Centroides de robot1, robot2, balón por frame |
| `output/analytics/IMG_9866_analytics.json` | Estadísticas completas del partido |
| `output/videos/IMG_9866_tracked.mp4` | Video con overlays de segmentación |
| `output/videos/IMG_9866_sidebyside.mp4` | Video side-by-side con vista cenital animada |
| `output/viz/IMG_9866/topdown_panel.png` | Panel heatmap + Voronoi + trails + stats |
| `output/viz/IMG_9866/heatmap_panel.png` | Heatmaps de actividad individual y combinado |
| `output/field_corners_IMG_9866.json` | Esquinas del campo para homografía |

---

## 8. Innovaciones técnicas

### 8.1 Ensamble YOLO + SAM3

Combinación de dos modelos complementarios:

- **YOLO** (`yolo26n.pt` entrenado en robots): detección rápida cada frame → bounding boxes
- **SAM3** (Meta AI): segmentación semántica de alta calidad → máscara precisa del balón

YOLO identifica a los robots aunque sean visualmente idénticos gracias al tracking por proximidad.  
SAM3 produce máscaras pixel-level del balón propagadas automáticamente a todo el video.

![Detección YOLO en frame 0: robot1 (azul), robot2 (naranja), balón (amarillo)](output/debug/IMG_9866_autodetect.jpg)

*YOLOv8 fine-tuned sobre ~48 imágenes anotadas de robots Zumo + balón naranja. Conf threshold = 0.25. Fallback HSV si YOLO no encuentra el balón.*

### 8.2 Tracking anti-swap con zona de fusión

Problema: cuando dos robots se tocan, los trackers simples intercambian las etiquetas.

Solución implementada:
1. **Regresión lineal** sobre historial de 12 frames para estimar velocidad
2. **Snapshot pre-fusión**: al detectar que los robots están a < 160 px, se guarda posición+velocidad antes del contacto
3. **Extrapolación durante el contacto**: las predicciones se calculan desde el snapshot, no desde la posición observada
4. **Hungarian assignment global** (scipy): asignación óptima detección↔label en cada frame

### 8.3 Predicción de tiro a gol

El sistema detecta tiros **antes** de que el balón entre a la portería:

1. Ajusta regresión lineal sobre los últimos 6 centroides del balón
2. Extrapola la trayectoria hasta 18 frames adelante
3. Si la trayectoria intersecta el bounding box de una portería (± 100 px), dispara evento `shot_on_goal`

Esto permite actuar cuando SAM3 pierde el balón al entrar a la portería.

### 8.4 Detección de porterías por mediana temporal

Las porterías son amarilla y azul, pero en el campo hay otros objetos de esos colores (robots con pequeños LEDs, ropa de personas, etc.).

Filtro de mediana temporal:
- Solo los píxeles que **permanecen estáticos** en la mayoría de los frames sobreviven
- Elimina robots en movimiento, personas, reflejos dinámicos
- Las porterías (fijas) generan máscaras estables

![Detección de porterías por HSV: azul (izquierda) y amarillo (derecha)](output/viz/IMG_9866/debug_goals.jpg)

*Resultado del filtro HSV sobre el frame real. Azul detecta la portería izquierda (a pesar del ruido de ropa), amarillo detecta la portería derecha con alta precisión.*

### 8.5 Vista cenital con homografía perspectiva

La cámara no está en posición cenital; tiene un ángulo de ~45°. Para corregirlo:

1. Se identifican las 4 esquinas del campo en coordenadas de imagen
2. Se calcula la matriz de homografía H con `cv2.findHomography`
3. Cada centroide se proyecta: `punto_campo = H × punto_camara`

Esto produce una vista top-down donde las posiciones son métricamente correctas aunque la cámara esté inclinada.

![Esquinas del campo con vista cenital preview](output/debug/debug_corners_IMG_9866.jpg)

*Las líneas verdes delimitan el campo. La miniatura en la esquina superior izquierda muestra la vista cenital resultante tras la homografía.*

---

## 9. Reproducibilidad

### Semillas y determinismo

- YOLO: modo inferencia (sin entrenamiento), determinista dado el mismo modelo
- SAM3: propagación determinista dado el mismo frame 0 y prompt
- Analytics: algoritmos puramente deterministas (sin muestreo aleatorio en inferencia)

### Dependencias principales

```
torch==2.5.1+cu126
torchvision==0.20.1+cu126
ultralytics>=8.3
opencv-python>=4.9
scipy>=1.12
numpy>=1.26
```

Ver `requirements.txt` para la lista completa.

### Cómo reproducir el resultado de IMG_9866

```bash
# 1. Activar entorno
source .venv/bin/activate
cd /home/uaqfif/Copa_FutBotMX_los_cuatro_frametasticos

# 2. Extraer frames (video en /mnt/d/videos/)
python scripts/extract_frames.py --video /mnt/d/videos/IMG_9866.MOV --step 3

# 3. Pipeline: YOLO robots + SAM3 balón
python scripts/pipeline.py --frames_dir output/frames/IMG_9866 --auto

# 4. Calibrar campo (esquinas del campo para homografía)
python scripts/auto_corners.py \
    --frame output/frames/IMG_9866/00000.jpg \
    --out   output/field_corners_IMG_9866.json

# 5. Analytics (con detección de gol por color HSV)
python scripts/analytics.py \
    --tracks     output/tracks/IMG_9866_tracks.json \
    --frames_dir output/frames/IMG_9866 \
    --fps 30 --step 3

# 6. Video side-by-side con vista cenital
python scripts/visualize.py video \
    --tracks     output/tracks/IMG_9866_tracks.json \
    --analytics  output/analytics/IMG_9866_analytics.json \
    --frames_dir output/frames/IMG_9866 \
    --corners    output/field_corners_IMG_9866.json \
    --output     output/videos/ --fps 10 --step 3

# 7. Visualizaciones estáticas
python scripts/visualize.py heatmap \
    --analytics output/analytics/IMG_9866_analytics.json \
    --bg        output/frames/IMG_9866/00000.jpg \
    --output    output/viz/IMG_9866/

python scripts/visualize.py topdown \
    --analytics output/analytics/IMG_9866_analytics.json \
    --corners   output/field_corners_IMG_9866.json \
    --output    output/viz/IMG_9866/
```