"""
Prepara el dataset YOLO a partir de anotaciones de anylabeling (JSON) o LabelImg (TXT).

1. Convierte JSON (anylabeling) → TXT (formato YOLO normalizado)
2. Descarta imágenes sin anotaciones o con shapes vacíos
3. Split 80% train / 20% val (reproducible con semilla fija)
4. Copia imágenes y labels a la estructura que espera YOLOv8
5. Genera dataset.yaml listo para entrenar

Uso:
    python scripts/prepare_yolo_dataset.py
    python scripts/prepare_yolo_dataset.py --classes robot ball  (orden = índice YOLO)
    python scripts/prepare_yolo_dataset.py --label_map bot:robot  (renombrar etiquetas)

Estructura de salida:
    output/yolo_dataset/
    ├── images/train/   ← JPGs de entrenamiento
    ├── images/val/     ← JPGs de validación
    ├── labels/train/   ← TXTs YOLO de entrenamiento
    ├── labels/val/     ← TXTs YOLO de validación
    └── dataset.yaml
"""

import argparse
import json
import os
import random
import shutil
import sys

import yaml


# ---------------------------------------------------------------------------
# Conversión JSON (anylabeling) → YOLO TXT
# ---------------------------------------------------------------------------

def json_to_yolo(json_path: str, class_to_idx: dict, label_map: dict) -> list[str]:
    """
    Convierte un JSON de anylabeling a líneas YOLO.
    Retorna lista de strings "class_idx cx cy w h" (normalizados 0-1).
    Retorna [] si no hay shapes válidos.
    """
    with open(json_path) as f:
        data = json.load(f)

    W = data.get("imageWidth", 0)
    H = data.get("imageHeight", 0)
    if W == 0 or H == 0:
        return []

    lines = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        pts = shape["points"]
        if len(pts) < 2:
            continue

        raw_label = shape["label"].strip()
        label = label_map.get(raw_label, raw_label)

        if label not in class_to_idx:
            continue

        x1, y1 = pts[0]
        x2, y2 = pts[1]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        cx = ((x1 + x2) / 2) / W
        cy = ((y1 + y2) / 2) / H
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H

        # Clamp por seguridad
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w  = max(0.001, min(1.0, w))
        h  = max(0.001, min(1.0, h))

        idx = class_to_idx[label]
        lines.append(f"{idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def find_annotation(img_path: str) -> tuple[str, str] | None:
    """
    Busca el archivo de anotación correspondiente a img_path.
    Retorna (annotation_path, format) donde format es "json" o "yolo".
    """
    base = os.path.splitext(img_path)[0]
    json_path = base + ".json"
    txt_path  = base + ".txt"
    if os.path.exists(json_path):
        return json_path, "json"
    if os.path.exists(txt_path):
        return txt_path, "yolo"
    return None


def scan_labels(images_dir: str, label_map: dict) -> set[str]:
    """Escanea todos los JSONs y devuelve el conjunto de etiquetas únicas encontradas."""
    labels = set()
    for fname in os.listdir(images_dir):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(images_dir, fname)) as f:
                data = json.load(f)
            for shape in data.get("shapes", []):
                raw = shape.get("label", "").strip()
                labels.add(label_map.get(raw, raw))
        except Exception:
            pass
    return labels


def copy_pair(img_src: str, txt_content: str, img_dst_dir: str, lbl_dst_dir: str):
    fname = os.path.basename(img_src)
    stem  = os.path.splitext(fname)[0]
    shutil.copy2(img_src, os.path.join(img_dst_dir, fname))
    with open(os.path.join(lbl_dst_dir, stem + ".txt"), "w") as f:
        f.write(txt_content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "output", "yolo_dataset"),
    )
    parser.add_argument(
        "--classes", nargs="+", default=[],
        help="Orden de clases (índice 0, 1, …). Default: auto-detectado.",
    )
    parser.add_argument(
        "--label_map", nargs="*", default=[],
        metavar="ORIG:NUEVO",
        help="Renombrar etiquetas, ej: bot:robot campo:field",
    )
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    images_dir  = os.path.join(dataset_dir, "images")

    if not os.path.isdir(images_dir):
        print(f"No existe: {images_dir}")
        sys.exit(1)

    # Parsear label_map
    label_map = {}
    for entry in args.label_map:
        if ":" not in entry:
            print(f"Formato inválido en --label_map: '{entry}'. Usa ORIG:NUEVO")
            sys.exit(1)
        orig, nuevo = entry.split(":", 1)
        label_map[orig.strip()] = nuevo.strip()

    # Detectar etiquetas únicas
    found_labels = scan_labels(images_dir, label_map)
    if not found_labels:
        print("No se encontraron etiquetas en los JSONs. ¿Ya anotaste algunas imágenes?")
        sys.exit(1)

    # Determinar orden de clases
    if args.classes:
        classes = args.classes
        unknown = found_labels - set(classes)
        if unknown:
            print(f"⚠  Etiquetas en JSONs no incluidas en --classes (se ignorarán): {unknown}")
    else:
        # Orden determinista: robot/bot primero, ball al final
        priority = ["robot", "ball"]
        classes = [c for c in priority if c in found_labels]
        classes += sorted(found_labels - set(classes))

    class_to_idx = {c: i for i, c in enumerate(classes)}

    print(f"Clases YOLO detectadas: {classes}")
    print(f"  label_map aplicado  : {label_map if label_map else '(ninguno)'}")
    print()

    # Recolectar pares (imagen, yolo_txt) con anotaciones válidas
    pairs = []  # (img_path, yolo_txt_str)
    skipped_no_ann = 0
    skipped_empty  = 0

    for fname in sorted(os.listdir(images_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMG_EXTS:
            continue

        img_path = os.path.join(images_dir, fname)
        ann = find_annotation(img_path)

        if ann is None:
            skipped_no_ann += 1
            continue

        ann_path, fmt = ann

        if fmt == "json":
            lines = json_to_yolo(ann_path, class_to_idx, label_map)
        else:
            with open(ann_path) as f:
                lines = [l.rstrip() for l in f if l.strip()]

        if not lines:
            skipped_empty += 1
            continue

        pairs.append((img_path, "\n".join(lines) + "\n"))

    print(f"Imágenes con anotaciones válidas : {len(pairs)}")
    print(f"Sin archivo de anotación         : {skipped_no_ann}")
    print(f"Anotación vacía (sin shapes)     : {skipped_empty}")
    print()

    if not pairs:
        print("No hay imágenes anotadas todavía. Termina de anotar y vuelve a correr.")
        sys.exit(0)

    # Split train / val
    random.seed(args.seed)
    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * args.val_ratio))
    val_set = pairs[:n_val]
    trn_set = pairs[n_val:]

    print(f"Train: {len(trn_set)}  |  Val: {len(val_set)}")
    print()

    # Crear carpetas destino
    for split in ("train", "val"):
        os.makedirs(os.path.join(dataset_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "labels", split), exist_ok=True)

    for img_path, txt in trn_set:
        copy_pair(img_path, txt,
                  os.path.join(dataset_dir, "images", "train"),
                  os.path.join(dataset_dir, "labels", "train"))

    for img_path, txt in val_set:
        copy_pair(img_path, txt,
                  os.path.join(dataset_dir, "images", "val"),
                  os.path.join(dataset_dir, "labels", "val"))

    # Generar dataset.yaml
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    cfg = {
        "path":  dataset_dir,
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(classes),
        "names": classes,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"dataset.yaml generado: {yaml_path}")
    print()
    print("Para entrenar:")
    print(f"  yolo train model=yolov8n.pt data={yaml_path} epochs=100 imgsz=640 batch=16")
    print()
    print("O desde Python:")
    print(f"  from ultralytics import YOLO")
    print(f"  YOLO('yolov8n.pt').train(data='{yaml_path}', epochs=100, imgsz=640, batch=16)")


if __name__ == "__main__":
    main()
