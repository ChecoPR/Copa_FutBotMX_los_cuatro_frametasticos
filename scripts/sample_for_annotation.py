"""
Extrae frames representativos de todos los videos para anotar en YOLO.

Estrategia: N frames por video muestreados en posiciones distribuidas
uniformemente (evita inicio/fin donde puede no haber acción).

Uso:
    python scripts/sample_for_annotation.py
    python scripts/sample_for_annotation.py --total 300 --videos_dir /mnt/c/videos
    python scripts/sample_for_annotation.py --per_video 3   # fijo por video
"""

import argparse
import math
import os
import sys

import cv2


def list_videos(videos_dir: str) -> list:
    exts = {".mov", ".mp4", ".avi", ".MOV", ".MP4"}
    paths = sorted(
        os.path.join(videos_dir, f)
        for f in os.listdir(videos_dir)
        if os.path.splitext(f)[1] in exts
    )
    return paths


def sample_frames_from_video(video_path: str, n: int, out_dir: str, prefix: str):
    """Extrae n frames distribuidos uniformemente del video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ⚠ No se pudo abrir: {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print(f"  ⚠ Frame count inválido: {video_path}")
        cap.release()
        return 0

    # Posiciones relativas evitando el primer y último 10%
    if n == 1:
        positions = [0.5]
    else:
        step = 0.8 / (n - 1)
        positions = [0.1 + i * step for i in range(n)]

    frame_indices = [max(0, min(total - 1, int(p * total))) for p in positions]

    saved = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = f"{prefix}_f{idx:05d}.jpg"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos_dir", default="/mnt/c/videos",
                        help="Carpeta con los videos")
    parser.add_argument("--out_dir",
                        default=os.path.join(os.path.dirname(__file__), "..",
                                             "output", "yolo_dataset", "images"),
                        help="Carpeta de salida para los frames")
    parser.add_argument("--total", type=int, default=250,
                        help="Total de frames a extraer (aprox)")
    parser.add_argument("--per_video", type=int, default=0,
                        help="Frames fijos por video (anula --total)")
    args = parser.parse_args()

    videos = list_videos(args.videos_dir)
    if not videos:
        print(f"No se encontraron videos en {args.videos_dir}")
        sys.exit(1)

    n_videos = len(videos)
    if args.per_video > 0:
        frames_per_video = args.per_video
    else:
        frames_per_video = max(2, math.ceil(args.total / n_videos))

    print(f"Videos encontrados : {n_videos}")
    print(f"Frames por video   : {frames_per_video}")
    print(f"Total estimado     : {n_videos * frames_per_video}")
    print(f"Carpeta de salida  : {args.out_dir}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    total_saved = 0
    for i, vpath in enumerate(videos):
        name = os.path.splitext(os.path.basename(vpath))[0]
        # Prefijo corto para nombres de archivo manejables
        prefix = name.replace("_singular_display", "")[:40]
        saved = sample_frames_from_video(vpath, frames_per_video, args.out_dir, prefix)
        total_saved += saved
        print(f"[{i+1:3d}/{n_videos}] {name[:50]:50s}  → {saved} frames")

    print(f"\nTotal guardado: {total_saved} frames en {args.out_dir}/")
    print("\nSiguientes pasos:")
    print("  1. Abre LabelImg en esa carpeta y anota 'robot' y 'ball'")
    print("     pip install labelImg && labelImg", args.out_dir)
    print("  2. Cuando termines, corre: python scripts/prepare_yolo_dataset.py")


if __name__ == "__main__":
    main()
