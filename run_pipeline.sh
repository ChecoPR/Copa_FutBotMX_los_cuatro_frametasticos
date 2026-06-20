#!/usr/bin/env bash
# run_pipeline.sh — Orquestador del pipeline completo de Copa FutBotMX.
#
# Ejecuta en secuencia las cinco etapas del pipeline:
#   1. Extracción de frames   (extract_frames.py)
#   2. Detección de esquinas  (auto_corners.py)
#   3. Tracking YOLO + SAM3   (pipeline.py --auto)
#   4. Analytics              (analytics.py)
#   5. Video side-by-side     (visualize.py video)
#
# Uso:
#   bash run_pipeline.sh --video <ruta.MOV> [opciones]
#
# Argumentos obligatorios:
#   --video <ruta>          Ruta al archivo de video (.MOV u otro formato soportado por OpenCV)
#
# Argumentos opcionales:
#   --step  <n>             Paso de extracción de frames (default: 1).
#                           Con step=3 y video a 30 fps → 10 fps efectivos.
#   --fps   <n>             FPS del video de salida (default: 10).
#   --ball_point <x,y>      Coordenadas manuales del centroide del balón en el frame de inicio.
#                           Anula la detección automática del balón (útil si YOLO falla).
#   --ball_frame <n>        Índice del frame donde inicializar el prompt de SAM3 para el balón
#                           (default: 0). Úsese con --ball_point cuando el balón no es visible
#                           en el frame 0.
#   --tl/--tr/--br/--bl <x,y>
#                           Esquinas del campo en coordenadas de imagen (top-left, top-right,
#                           bottom-right, bottom-left). Si se especifican los 4 puntos se
#                           omite la detección automática de esquinas.
#   --skip_tracking         Omite las etapas 1–3 y reutiliza el tracks JSON existente.
#   --skip_corners          Omite la detección de esquinas si el JSON ya existe.
#   --skip_video            Omite el render del video side-by-side (etapa 5).
#   --open                  Abre el video resultante en el explorador de Windows (WSL2).
#   --python <ruta>         Intérprete Python a usar (default: .venv/bin/python3).
#
# Salidas generadas en output/:
#   tracks_<VIDEO>_tracks.json        Centroides y bounding boxes por frame
#   analytics_<VIDEO>_analytics.json  Posesión, velocidad, eventos y goles
#   field_corners_<VIDEO>.json        Esquinas del campo para homografía
#   videos/<VIDEO>_sidebyside.mp4     Video side-by-side con HUD de estadísticas
#
# Ejemplos:
#   bash run_pipeline.sh --video videos/IMG_9866.MOV
#   bash run_pipeline.sh --video videos/IMG_9869.MOV --step 3 --fps 10
#   bash run_pipeline.sh --video videos/IMG_9866.MOV --ball_point 1085,267 --ball_frame 5
#   bash run_pipeline.sh --video videos/IMG_9866.MOV \
#       --tl 90,0 --tr 1775,5 --br 1875,575 --bl 290,470
#   bash run_pipeline.sh --video videos/IMG_9866.MOV --skip_tracking
#   bash run_pipeline.sh --video videos/IMG_9866.MOV --skip_video

set -euo pipefail

# ── Colores ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

step() { echo -e "\n${CYAN}${BOLD}▶ $*${NC}"; }
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠ $*${NC}"; }
die()  { echo -e "${RED}✗ $*${NC}"; exit 1; }

# ── Defaults ─────────────────────────────────────────────────────────────────
VIDEO=""
STEP=1
FPS=10
BALL_POINT=""
BALL_FRAME=0
CORNER_TL=""; CORNER_TR=""; CORNER_BR=""; CORNER_BL=""
SKIP_TRACKING=false
SKIP_CORNERS=false
SKIP_VIDEO=false
OPEN_RESULT=false
PYTHON=".venv/bin/python3"

# ── Parseo de argumentos ─────────────────────────────────────────────────────
_arg() {
    # Valida que el siguiente token exista y no sea otro flag
    [[ $# -ge 2 && "${2:-}" != --* ]] || die "Falta valor para $1  (ej: $1 valor)"
    echo "$2"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)         VIDEO="$(_arg "$@")";       shift 2 ;;
        --step)          STEP="$(_arg "$@")";         shift 2 ;;
        --fps)           FPS="$(_arg "$@")";          shift 2 ;;
        --ball_point)    BALL_POINT="$(_arg "$@")";   shift 2 ;;
        --ball_frame)    BALL_FRAME="$(_arg "$@")";   shift 2 ;;
        --tl)            CORNER_TL="$(_arg "$@")";    shift 2 ;;
        --tr)            CORNER_TR="$(_arg "$@")";    shift 2 ;;
        --br)            CORNER_BR="$(_arg "$@")";    shift 2 ;;
        --bl)            CORNER_BL="$(_arg "$@")";    shift 2 ;;
        --skip_tracking) SKIP_TRACKING=true; shift ;;
        --skip_corners)  SKIP_CORNERS=true;  shift ;;
        --skip_video)    SKIP_VIDEO=true;    shift ;;
        --open)          OPEN_RESULT=true;   shift ;;
        --python)        PYTHON="$(_arg "$@")"; shift 2 ;;
        -h|--help)
            sed -n '/^# Uso/,/^[^#]/p' "$0" | head -12
            exit 0
            ;;
        *)
            die "Argumento desconocido: $1\n  Usa --help para ver opciones."
            ;;
    esac
done

# ── Validación básica ─────────────────────────────────────────────────────────
[[ -z "$VIDEO" ]] && die "Falta --video  (ej: --video videos/IMG_9866.MOV)"
[[ -f "$VIDEO" ]] || die "Video no encontrado: $VIDEO"
$PYTHON -c "import cv2" 2>/dev/null || die "cv2 no disponible en $PYTHON\n  Activa el venv: source .venv/bin/activate"

# ── Rutas derivadas del nombre del video ─────────────────────────────────────
VNAME=$(basename "$VIDEO" | sed 's/\.[^.]*$//')   # IMG_9866
FRAMES_DIR="output/frames/${VNAME}"
OUTPUT_DIR="output"
TRACKS_JSON="output/tracks_${VNAME}.json"
ANALYTICS_JSON="output/analytics_${VNAME}.json"
CORNERS_JSON="output/field_corners_${VNAME}.json"
CORNER_FRAME="${FRAMES_DIR}/00000.jpg"

echo -e "${BOLD}═══════════════════════════════════════════${NC}"
echo -e "${BOLD}  Copa FutBotMX — Pipeline completo${NC}"
echo -e "${BOLD}═══════════════════════════════════════════${NC}"
echo "  Video  : $VIDEO"
echo "  Nombre : $VNAME"
echo "  Step   : $STEP   FPS salida: $FPS"
echo "  Frames : $FRAMES_DIR"
[[ -n "$BALL_POINT" ]] && echo "  Balón  : $BALL_POINT (frame $BALL_FRAME)"
[[ -n "$CORNER_TL" ]] && echo "  Esquinas: manual ($CORNER_TL | $CORNER_TR | $CORNER_BR | $CORNER_BL)"

# ════════════════════════════════════════════════════════════════════════════
# PASO 1 — Extracción de frames
# ════════════════════════════════════════════════════════════════════════════
step "PASO 1/5 — Extraer frames (step=$STEP)"

if [[ -d "$FRAMES_DIR" ]] && [[ $(ls "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l) -gt 0 ]]; then
    N_EXISTING=$(ls "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l)
    warn "Ya existen $N_EXISTING frames en $FRAMES_DIR — saltando extracción"
    warn "  (borra la carpeta para volver a extraer)"
else
    $PYTHON scripts/extract_frames.py \
        --video "$VIDEO" \
        --output_dir "$FRAMES_DIR" \
        --step "$STEP"
fi

N_FRAMES=$(ls "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l)
ok "$N_FRAMES frames en $FRAMES_DIR"

if [[ $N_FRAMES -gt 300 ]]; then
    warn "⚠  $N_FRAMES frames es mucho para SAM3 — puede morir por OOM."
    warn "   Recomendado: borra output/frames/${VNAME}/ y usa --step 3 o --step 5"
    warn "   (con step=3 quedan ~$((N_FRAMES/3)) frames)"
fi

# ════════════════════════════════════════════════════════════════════════════
# PASO 2 — Detección de esquinas del campo
# ════════════════════════════════════════════════════════════════════════════
step "PASO 2/5 — Esquinas del campo"

if $SKIP_CORNERS && [[ -f "$CORNERS_JSON" ]]; then
    ok "Usando esquinas existentes: $CORNERS_JSON"

elif [[ -n "$CORNER_TL" && -n "$CORNER_TR" && -n "$CORNER_BR" && -n "$CORNER_BL" ]]; then
    # Modo manual
    $PYTHON scripts/auto_corners.py \
        --frame "$CORNER_FRAME" \
        --tl "$CORNER_TL" --tr "$CORNER_TR" \
        --br "$CORNER_BR" --bl "$CORNER_BL" \
        --out "$CORNERS_JSON"
    ok "Esquinas manuales guardadas en $CORNERS_JSON"

elif [[ -f "$CORNERS_JSON" ]]; then
    ok "Reutilizando $CORNERS_JSON (usa --skip_corners para forzar)"

else
    # Modo automático
    if $PYTHON scripts/auto_corners.py \
        --frame "$CORNER_FRAME" \
        --out "$CORNERS_JSON" 2>&1; then
        ok "Esquinas detectadas automáticamente: $CORNERS_JSON"
    else
        GRID_PATH="output/grid_${VNAME}.jpg"
        warn "Detección automática falló."
        warn "Generando cuadrícula de coordenadas: $GRID_PATH"
        $PYTHON scripts/auto_corners.py \
            --frame "$CORNER_FRAME" \
            --grid "$GRID_PATH" \
            --out "$CORNERS_JSON" || true
        echo ""
        echo -e "${YELLOW}══════════════════════════════════════════════════${NC}"
        echo -e "${YELLOW}  Abre la cuadricula en Windows y lee las esquinas:${NC}"
        echo -e "${YELLOW}  explorer.exe \"$(wslpath -w "$(pwd)/$GRID_PATH")\"${NC}"
        echo -e "${YELLOW}  Luego vuelve a correr con:${NC}"
        echo -e "${YELLOW}  bash run_pipeline.sh --video $VIDEO \\${NC}"
        echo -e "${YELLOW}      --tl X,Y --tr X,Y --br X,Y --bl X,Y${NC}"
        echo -e "${YELLOW}══════════════════════════════════════════════════${NC}"
        exit 1
    fi
fi

# ════════════════════════════════════════════════════════════════════════════
# PASO 3 — Tracking (YOLO + SAM3)
# ════════════════════════════════════════════════════════════════════════════
step "PASO 3/5 — Tracking YOLO + SAM3"

if $SKIP_TRACKING && [[ -f "$TRACKS_JSON" ]]; then
    ok "Reutilizando tracks: $TRACKS_JSON"
else
    PIPELINE_ARGS=(
        --frames_dir "$FRAMES_DIR"
        --output_dir "$OUTPUT_DIR"
        --fps "$FPS"
        --auto
    )
    [[ -n "$BALL_POINT" ]] && PIPELINE_ARGS+=(--ball_point "$BALL_POINT" --ball_frame "$BALL_FRAME")

    $PYTHON scripts/pipeline.py "${PIPELINE_ARGS[@]}"

    # pipeline.py guarda en output/tracks/<VNAME>_tracks.json
    PIPELINE_OUT="output/tracks/${VNAME}_tracks.json"
    if [[ -f "$PIPELINE_OUT" ]]; then
        cp "$PIPELINE_OUT" "$TRACKS_JSON"
    fi
    ok "Tracks: $TRACKS_JSON"
fi

# ════════════════════════════════════════════════════════════════════════════
# PASO 4 — Analytics
# ════════════════════════════════════════════════════════════════════════════
step "PASO 4/5 — Analytics"

$PYTHON scripts/analytics.py \
    --tracks     "$TRACKS_JSON" \
    --output     "$ANALYTICS_JSON" \
    --fps        30 \
    --step       "$STEP" \
    --frames_dir "$FRAMES_DIR"

ok "Analytics: $ANALYTICS_JSON"

# ════════════════════════════════════════════════════════════════════════════
# PASO 5 — Video side-by-side con HUD
# ════════════════════════════════════════════════════════════════════════════
step "PASO 5/5 — Video side-by-side"

if $SKIP_VIDEO; then
    warn "Saltando render de video (--skip_video)"
else
    VIDEO_ARGS=(
        video
        --tracks     "$TRACKS_JSON"
        --analytics  "$ANALYTICS_JSON"
        --frames_dir "$FRAMES_DIR"
        --output     "output/videos/"
        --fps        "$FPS"
        --step       "$STEP"
    )
    [[ -f "$CORNERS_JSON" ]] && VIDEO_ARGS+=(--corners "$CORNERS_JSON")

    $PYTHON scripts/visualize.py "${VIDEO_ARGS[@]}"

    VIDEO_OUT=$(ls output/videos/${VNAME}*.mp4 2>/dev/null | sort | tail -1 || true)
    if [[ -n "$VIDEO_OUT" ]]; then
        ok "Video: $VIDEO_OUT"
        if $OPEN_RESULT; then
            explorer.exe "$(wslpath -w "$(pwd)/$VIDEO_OUT")" 2>/dev/null || true
        else
            echo ""
            echo -e "  Abrir en Windows:"
            echo -e "  ${CYAN}explorer.exe \"$(wslpath -w "$(pwd)/$VIDEO_OUT")\"${NC}"
        fi
    fi
fi

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  Pipeline completo ✓${NC}"
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════${NC}"
echo "  Tracks    : $TRACKS_JSON"
echo "  Analytics : $ANALYTICS_JSON"
echo "  Esquinas  : $CORNERS_JSON"
[[ -n "${VIDEO_OUT:-}" ]] && echo "  Video     : $VIDEO_OUT"
