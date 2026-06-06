# Copa FutBotMX — Los Cuatro Frametásticos
## Capítulo Visión por Computadora · Secihti 2026

**Categoría:** Profesional  
**Deadline GitHub:** 19 de junio de 2026 · 23:59 hrs  
**Equipo:** Los Cuatro Frametásticos  

---

## Expertise del equipo

- Metaheurísticas (algoritmos evolutivos, optimización por enjambre)
- Deep learning (clasificación, detección, segmentación)
- Modelos ensamblados (stacking, boosting, voting)

Esto nos coloca en **Categoría Profesional**. La rúbrica exige innovación sobre SAM3, métricas cuantitativas y resultados reproducibles.

---

## Arquitectura general

```
Videos .MOV (98 videos, 1920×1080)
        │
        ▼
┌─────────────────────────────────────────┐
│  ETAPA 1 — EXTRACCIÓN DE FRAMES         │
│  scripts/extract_frames.py              │
│  · Paso configurable (default: step=3)  │
│  · Salida: output/frames/<video>/       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  ETAPA 2 — SEGMENTACIÓN Y TRACKING      │
│  scripts/pipeline.py                    │
│                                         │
│  SAM3 (Meta AI) — base obligatoria      │
│  ├── Texto prompt → "ball" funciona     │
│  ├── Box prompt → objeto visual (1 caja)│
│  └── Tracker points → objetos adicional │
│                                         │
│  Salidas:                               │
│  · output/masks/<video>/*.png           │
│  · output/tracks/<video>_tracks.json   │
│  · output/videos/<video>_tracked.mp4   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  ETAPA 3 — CLASIFICACIÓN DE EQUIPO      │
│  scripts/classify_teams.py  [TODO]      │
│                                         │
│  · Análisis de color HSV en máscaras    │
│  · K-Means / GMM sobre features de      │
│    color por robot → equipo A vs B      │
│  · Actualiza tracks con "team" label    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  ETAPA 4 — ANALYTICS ENGINE             │
│  scripts/analytics.py  [TODO]           │
│                                         │
│  · Posesión: proximidad robot↔balón     │
│    + contacto (intersección de máscaras)│
│  · Velocidad: |Δcentroide| / Δt         │
│  · Detección de eventos:                │
│    - Pase: cambio de posesión           │
│    - Tiro a gol: balón cerca del arco   │
│    - Colisión: robots con IOU > umbral  │
│  · Mapa de zonas: diagramas de Voronoi  │
│    por posición instantánea de robots   │
│                                         │
│  INNOVACIÓN — Metaheurísticas:          │
│  · PSO/GA para optimizar umbrales de    │
│    detección de eventos (posesión,      │
│    colisión) sobre ground truth manual  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  ETAPA 5 — VISUALIZACIÓN                │
│  scripts/visualize.py  [TODO]           │
│                                         │
│  · Heatmaps de actividad por equipo     │
│  · Trails de trayectorias               │
│  · Diagrama de Voronoi dinámico         │
│  · Gráficas de posesión temporal        │
│  · Video final: original | segmentado   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  ETAPA 6 — INNOVACIÓN AVANZADA  [TODO]  │
│                                         │
│  Opción A — Fine-tuning SAM3:           │
│  · Anotar ~200 frames con robots+balón  │
│  · Fine-tune el decoder de SAM3         │
│  · Medir IoU antes/después              │
│                                         │
│  Opción B — Ensamble SAM3 + detector:   │
│  · YOLOv8-nano (rápido) para detectar   │
│    robots → bounding boxes como         │
│    prompts de SAM3 automáticamente      │
│  · Elimina necesidad de clic manual     │
│  · SAM3 produce máscaras precisas,      │
│    YOLO produce boxes iniciales         │
│                                         │
│  Opción C — ByteTrack integration:      │
│  · SAM3 genera máscaras por frame       │
│  · ByteTrack mantiene IDs consistentes  │
│    incluso con oclusiones               │
└─────────────────────────────────────────┘
```

---

## Estado actual (5 junio 2026)

### ✅ Completado
- Entorno configurado (Python 3.12, PyTorch 2.12, CUDA 12.6)
- SAM3 descargado y funcionando (parches para Maxwell sm_52)
- Pipeline básico operativo: extracción, segmentación, tracking, máscaras, JSON, video
- Herramienta interactiva `pick_points.py` para seleccionar objetos con clic
- Test exitoso en IMG_9866.MOV (102 frames, 4 objetos detectados)

### 🔄 En progreso
- Propagación del video de prueba con prompts de punto (robot1 box + robot2/ball tracker)

### 📋 Pendiente
- Clasificación de equipos por color
- Analytics: posesión, velocidad, eventos
- Visualizaciones: heatmaps, Voronoi, trails
- Procesamiento batch de los 98 videos
- Innovación avanzada (ensamble o fine-tuning)
- Video demo 2 min + Reel Instagram

---

## Constraints técnicas importantes (GPU)

| GPU | VRAM | Compute | Notas |
|-----|------|---------|-------|
| GTX 1080 (GPU 0) | 8 GB | sm_61 | Triton OK |
| TITAN X (GPU 1) | 11 GB | sm_52 | En uso; bfloat16→float16; Triton→scipy |

**WSL2**: NCCL multi-GPU falla → siempre `gpus_to_use=[1]`

**Parches aplicados a SAM3:**
- `sam3/sam3/model/sam3_base_predictor.py:205` — float16
- `sam3/sam3/model/sam3_video_inference.py:800,908` — float16
- `sam3/sam3/model/sam3_tracking_predictor.py:50,1097,1148` — float16
- `sam3/sam3/model/sam3_image.py:864` — float16
- `sam3/sam3/perflib/connected_components.py` — scipy CPU fallback

---

## Limitaciones de la API SAM3 (documentadas)

1. `add_prompt(text/box)` llama `reset_state` — solo la última llamada survives
2. Box prompt sin texto: **solo 1 caja** permitida (visual prompt mode)
3. Point prompt: requiere `previous_stages_out` inicializado (necesita caja/texto previo)
4. **Patrón que funciona:**
   ```python
   add_prompt(bounding_boxes=[[x,y,w,h]], bounding_box_labels=[1])  # objeto 1
   add_prompt(points=[[cx,cy]], point_labels=[1], obj_id=N)          # objeto 2+
   ```
5. Coordenadas de caja: normalizadas [0,1] como [xmin, ymin, width, height]

---

## Comandos clave

```bash
# Activar entorno
source .venv/bin/activate
cd /home/uaqfif/Copa_FutBotMX_los_cuatro_frametasticos

# Extraer frames de un video
python scripts/extract_frames.py --video /mnt/d/videos/IMG_9866.MOV --step 3

# Seleccionar puntos interactivamente
python scripts/pick_points.py --frames_dir output/frames/IMG_9866

# Correr pipeline con puntos
python scripts/pipeline.py \
  --frames_dir output/frames/IMG_9866 \
  --point_prompts "robot1:850,306 robot2:1192,185 ball:1083,263"

# Correr pipeline con texto (solo balón funciona bien)
python scripts/pipeline.py \
  --frames_dir output/frames/IMG_9866 \
  --prompts "ball"
```

---

## Roadmap de entregables (hacia el 19 jun)

| Semana | Prioridad | Tarea |
|--------|-----------|-------|
| 5–8 jun | 🔴 Alta | Pipeline completo: máscaras + tracks de todos los objetos |
| 5–8 jun | 🔴 Alta | Clasificación de equipos por color HSV |
| 9–11 jun | 🔴 Alta | Analytics: posesión, velocidad, eventos clave |
| 9–11 jun | 🟡 Media | Procesamiento batch de los 98 videos |
| 12–14 jun | 🔴 Alta | Visualizaciones: heatmaps, Voronoi, trails |
| 12–14 jun | 🟡 Media | Innovación avanzada: YOLO→SAM3 ensamble o fine-tuning |
| 15–17 jun | 🔴 Alta | Video demo 2 min (original + segmentado) |
| 15–17 jun | 🔴 Alta | README.md completo con métricas y reproducibilidad |
| 18–19 jun | 🔴 Alta | Reel Instagram (30 seg highlights) + link en README |
| 18–19 jun | 🟡 Media | Revisión final, limpieza del repo |

---

## Criterios de evaluación profesional a cubrir

- [x] Pipeline funcional con SAM3
- [ ] **Innovación**: ensamble SAM3+YOLO o fine-tuning
- [ ] **Métricas cuantitativas**: IoU, velocidad (px/s), % posesión
- [ ] Visualizaciones (heatmap + al menos una más)
- [ ] Video demo ≤ 2 min
- [ ] Reel Instagram ≥ 30 seg
- [ ] README con instrucciones reproducibles
- [ ] Resultados reproducibles (semillas fijas, reqs documentados)

---

## Estructura del repositorio

```
Copa_FutBotMX_los_cuatro_frametasticos/
├── CLAUDE.md                  # Este archivo
├── README.md                  # Entregable final (sección 3.5.4)
├── requirements.txt
├── scripts/
│   ├── extract_frames.py      # Extracción de frames
│   ├── pick_points.py         # Selección interactiva de puntos
│   ├── pipeline.py            # Pipeline SAM3 principal
│   ├── classify_teams.py      # [TODO] Clasificación de equipos
│   ├── analytics.py           # [TODO] Posesión, velocidad, eventos
│   └── visualize.py           # [TODO] Heatmaps, Voronoi, video final
├── sam3/                      # SAM3 repo (submodule o local)
├── output/
│   ├── frames/                # Frames extraídos (no versionar)
│   ├── masks/                 # Máscaras binarias (no versionar)
│   ├── tracks/                # JSON de tracks (versionar muestras)
│   └── videos/                # Videos procesados (versionar demo)
└── notebooks/
    └── analysis.ipynb         # [TODO] Análisis exploratorio
```
