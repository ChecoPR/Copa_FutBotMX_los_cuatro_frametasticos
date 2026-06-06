"""
Extrae frames de un video .MOV para procesamiento con SAM3.

Uso:
    python extract_frames.py --video /mnt/d/videos/IMG_9866.MOV --step 3
    python extract_frames.py --video /mnt/d/videos/IMG_9866.MOV --step 1 --max_frames 300
"""

import argparse
import os
import cv2


def extract_frames(video_path: str, output_dir: str, step: int = 3, max_frames: int = None) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            filename = os.path.join(output_dir, f"{saved:05d}.jpg")
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            if max_frames and saved >= max_frames:
                break
        frame_idx += 1

    cap.release()

    info = {
        "video": video_path,
        "output_dir": output_dir,
        "fps": fps,
        "resolution": f"{w}x{h}",
        "total_frames_video": total,
        "frames_extracted": saved,
        "step": step,
        "effective_fps": fps / step,
    }

    print(f"Video     : {os.path.basename(video_path)}")
    print(f"Resolución: {w}x{h}  |  FPS: {fps:.1f}")
    print(f"Frames totales en video : {total}")
    print(f"Frames extraídos (step={step}): {saved}")
    print(f"FPS efectivo: {info['effective_fps']:.1f}")
    print(f"Guardados en: {output_dir}")

    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Ruta al video .MOV")
    parser.add_argument("--output_dir", default=None, help="Directorio de salida (default: output/frames/<nombre_video>)")
    parser.add_argument("--step", type=int, default=3, help="Guardar 1 de cada N frames (default: 3)")
    parser.add_argument("--max_frames", type=int, default=None, help="Límite de frames a extraer")
    args = parser.parse_args()

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "output", "frames", video_name
    )

    extract_frames(args.video, output_dir, step=args.step, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
