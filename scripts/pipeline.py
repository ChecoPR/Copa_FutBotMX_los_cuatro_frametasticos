"""
Pipeline principal: segmentación y tracking de fútbol robótico con SAM3.

Uso con prompts de texto:
    python scripts/pipeline.py --frames_dir output/frames/IMG_9866 --prompts "robot,ball"

Uso con prompts de punto (más confiable para objetos poco comunes):
    python scripts/pipeline.py --frames_dir output/frames/IMG_9866 \
        --point_prompts "robot:540,630 robot:1200,390 ball:1080,510"

    Formato: "label:x,y" separados por espacio.
    Cada entrada crea un objeto independiente en el frame 0.
    Las coordenadas son píxeles absolutos (x desde izquierda, y desde arriba).

Se pueden combinar ambos:
    --prompts "field" --point_prompts "robot:540,630 ball:1080,510"

Salida:
    - output/videos/<nombre>_tracked.mp4
    - output/tracks/<nombre>_tracks.json
    - output/masks/<nombre>/<frame>_obj<id>.png  (máscaras binarias, 0/255)
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SAM3_REPO = os.path.join(os.path.dirname(__file__), "..", "sam3")
BPE_PATH = os.path.join(SAM3_REPO, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

LABEL_COLORS = {
    "robot":      (255, 100,   0),
    "robot1":     (255, 100,   0),
    "robot2":     (  0, 100, 255),
    "ball":       (  0, 255, 255),
    "robot red":  (  0,   0, 255),
    "robot blue": (255,   0,   0),
    "field":      (  0, 180,   0),
}
DEFAULT_COLOR = (200, 200, 200)


def build_predictor():
    from sam3.model_builder import build_sam3_video_predictor
    gpus = [1]  # TITAN X (11GB)
    print(f"Cargando SAM3 en GPU(s): {gpus}")
    predictor = build_sam3_video_predictor(bpe_path=BPE_PATH, gpus_to_use=gpus)
    print("Predictor listo.")
    return predictor


def get_frame_paths(frames_dir: str) -> list:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])


def parse_point_prompts(raw: str) -> list:
    """
    Parsea el argumento --point_prompts.
    Formato: "label:x,y label:x,y ..."
    Devuelve lista de (label, x, y).
    """
    entries = []
    for token in raw.strip().split():
        if ":" not in token:
            raise ValueError(f"Formato inválido en point_prompt '{token}'. Usa label:x,y")
        label, coords = token.split(":", 1)
        parts = coords.split(",")
        if len(parts) != 2:
            raise ValueError(f"Coordenadas inválidas en '{token}'. Usa label:x,y")
        x, y = float(parts[0]), float(parts[1])
        entries.append((label.strip(), x, y))
    return entries


def mask_to_centroid(mask_np: np.ndarray):
    ys, xs = np.where(mask_np)
    if len(xs) == 0:
        return None
    return (int(xs.mean()), int(ys.mean()))


def render_overlay(frame_bgr: np.ndarray, outputs: dict, alpha: float = 0.5) -> np.ndarray:
    overlay = frame_bgr.copy()
    for obj_id, obj_data in outputs.items():
        mask = obj_data.get("mask")
        label = obj_data.get("label", f"obj_{obj_id}")
        if mask is None:
            continue
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask = mask.astype(bool)
        color = LABEL_COLORS.get(label, DEFAULT_COLOR)
        overlay[mask] = (overlay[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        centroid = mask_to_centroid(mask)
        if centroid:
            cv2.circle(overlay, centroid, 6, color, -1)
            cv2.putText(overlay, f"{label}[{obj_id}]", (centroid[0] + 8, centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return overlay


def run_pipeline(
    frames_dir: str,
    text_prompts: list,
    point_prompts: list,   # lista de (label, x, y) en píxeles absolutos
    output_dir: str,
    fps: float = 10.0,
):
    frame_paths = get_frame_paths(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No hay frames en {frames_dir}")
    print(f"Frames encontrados: {len(frame_paths)}")

    # Dimensiones del frame para normalizar coordenadas
    sample = cv2.imread(frame_paths[0])
    frame_H, frame_W = sample.shape[:2]

    predictor = build_predictor()

    resp = predictor.handle_request({"type": "start_session", "resource_path": frames_dir})
    session_id = resp["session_id"]
    print(f"Sesión iniciada: {session_id}")

    obj_id_to_label = {}

    # add_prompt llama reset_state en cada llamada de texto/caja.
    # Estrategia:
    #   - Texto solo → una llamada por prompt (solo la última queda activa, issue conocido)
    #   - Puntos → primer clic = caja (inicializa SAM3), resto = tracker points (sin reset)
    #   - Combinado → texto primero, luego tracker points

    if point_prompts:
        # --- Primer punto → caja normalizada (inicializa previous_stages_out) ---
        first_label, first_cx, first_cy = point_prompts[0]
        box_half_w = 60 / frame_W
        box_half_h = 60 / frame_H
        xmin = max(0.0, (first_cx / frame_W) - box_half_w)
        ymin = max(0.0, (first_cy / frame_H) - box_half_h)
        w    = min(2 * box_half_w, 1.0 - xmin)
        h    = min(2 * box_half_h, 1.0 - ymin)
        print(f"  Caja prompt (1er objeto): '{first_label}' @ ({int(first_cx)},{int(first_cy)})")

        r = predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "bounding_boxes": [[xmin, ymin, w, h]],
            "bounding_box_labels": [1],
        })
        if r and r.get("outputs") and r["outputs"].get("out_obj_ids") is not None:
            for oid in r["outputs"]["out_obj_ids"]:
                obj_id_to_label[int(oid)] = first_label

        # --- Resto de puntos → tracker (no llama reset_state) ---
        next_obj_id = max(obj_id_to_label.keys(), default=-1) + 1
        for label, cx, cy in point_prompts[1:]:
            print(f"  Tracker point: '{label}' @ ({int(cx)},{int(cy)})  obj_id={next_obj_id}")
            predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "points": [[float(cx), float(cy)]],
                "point_labels": [1],
                "obj_id": next_obj_id,
                "rel_coordinates": False,
            })
            obj_id_to_label[next_obj_id] = label
            next_obj_id += 1

    elif text_prompts:
        # --- Solo texto ---
        for prompt in text_prompts:
            print(f"  Texto prompt: '{prompt}'")
            r = predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": prompt,
            })
            if r and r.get("outputs") and r["outputs"].get("out_obj_ids") is not None:
                for oid in r["outputs"]["out_obj_ids"]:
                    obj_id_to_label[int(oid)] = prompt

    print(f"  Objetos registrados: {obj_id_to_label}")

    # --- Propagación ---
    print("Propagando en video...")
    outputs_per_frame = {}
    for response in predictor.handle_stream_request({
        "type": "propagate_in_video",
        "session_id": session_id,
    }):
        fidx = response["frame_index"]
        raw_out = response["outputs"]
        if raw_out is None:
            continue

        obj_ids = raw_out.get("out_obj_ids", [])
        probs   = raw_out.get("out_probs", [])
        masks   = raw_out.get("out_binary_masks", [])

        labeled = {}
        for i, oid in enumerate(obj_ids):
            oid_int = int(oid)
            labeled[oid_int] = {
                "mask":  masks[i] if i < len(masks) else None,
                "score": float(probs[i]) if i < len(probs) else 0.0,
                "label": obj_id_to_label.get(oid_int, f"obj_{oid_int}"),
            }
        outputs_per_frame[fidx] = labeled

        if fidx % 20 == 0:
            n = len([v for v in labeled.values() if v["mask"] is not None])
            print(f"  Frame {fidx}: {n} objetos")

    predictor.handle_request({"type": "close_session", "session_id": session_id})
    print(f"Total frames procesados: {len(outputs_per_frame)}")

    name = os.path.basename(frames_dir.rstrip("/"))

    # --- Guardar máscaras binarias ---
    masks_dir = os.path.join(output_dir, "masks", name)
    os.makedirs(masks_dir, exist_ok=True)
    for fidx, objs in outputs_per_frame.items():
        for obj_id, obj_data in objs.items():
            mask = obj_data["mask"]
            if mask is None:
                continue
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            mask_png = mask.astype(bool).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(masks_dir, f"{fidx:05d}_obj{obj_id}.png"), mask_png)
    print(f"Máscaras guardadas: {masks_dir}/")

    # --- Guardar tracks JSON ---
    tracks_dir = os.path.join(output_dir, "tracks")
    os.makedirs(tracks_dir, exist_ok=True)
    tracks_json = {}
    for fidx, objs in outputs_per_frame.items():
        tracks_json[str(fidx)] = {
            str(obj_id): {
                "centroid": mask_to_centroid(
                    obj_data["mask"].cpu().numpy() if isinstance(obj_data["mask"], torch.Tensor)
                    else obj_data["mask"]
                ) if obj_data["mask"] is not None else None,
                "score":  obj_data["score"],
                "label":  obj_data["label"],
            }
            for obj_id, obj_data in objs.items()
        }
    tracks_path = os.path.join(tracks_dir, f"{name}_tracks.json")
    with open(tracks_path, "w") as f:
        json.dump(tracks_json, f, indent=2)
    print(f"Tracks guardados: {tracks_path}")

    # --- Renderizar video ---
    sample = cv2.imread(frame_paths[0])
    h, w = sample.shape[:2]
    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    video_out = os.path.join(videos_dir, f"{name}_tracked.mp4")
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fidx, fpath in enumerate(frame_paths):
        frame = cv2.imread(fpath)
        if fidx in outputs_per_frame:
            frame = render_overlay(frame, outputs_per_frame[fidx])
        writer.write(frame)
    writer.release()
    print(f"Video guardado: {video_out}")

    return tracks_json


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--prompts", default="",
                        help="Prompts de texto separados por coma")
    parser.add_argument("--point_prompts", default="",
                        help='Prompts de punto. Formato: "label:x,y label:x,y ..."')
    parser.add_argument("--output_dir",
                        default=os.path.join(os.path.dirname(__file__), "..", "output"))
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    text_prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    point_prompts = parse_point_prompts(args.point_prompts) if args.point_prompts.strip() else []

    if not text_prompts and not point_prompts:
        parser.error("Especifica al menos --prompts o --point_prompts")

    if text_prompts:
        print(f"Texto prompts: {text_prompts}")
    if point_prompts:
        print(f"Punto prompts: {point_prompts}")

    run_pipeline(args.frames_dir, text_prompts, point_prompts, args.output_dir, args.fps)


if __name__ == "__main__":
    main()
