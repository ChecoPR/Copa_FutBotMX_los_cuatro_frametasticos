"""
Verifica que todo el entorno esté listo antes de correr el pipeline.
Corre: python scripts/test_setup.py
"""

import sys
import os

def check(label, fn):
    try:
        result = fn()
        print(f"  [OK] {label}" + (f": {result}" if result else ""))
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False

print("\n=== Verificación del entorno ===\n")

checks = [
    ("Python >= 3.10", lambda: sys.version),
    ("torch", lambda: __import__("torch").__version__),
    ("CUDA disponible", lambda: str(__import__("torch").cuda.is_available())),
    ("GPU", lambda: __import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else "CPU only"),
    ("opencv-python", lambda: __import__("cv2").__version__),
    ("numpy", lambda: __import__("numpy").__version__),
    ("sam3", lambda: __import__("sam3").__version__ if hasattr(__import__("sam3"), "__version__") else "instalado"),
    ("huggingface_hub", lambda: __import__("huggingface_hub").__version__),
    ("HF token", lambda: __import__("huggingface_hub").get_token() or "NO TOKEN"),
    ("Videos en /mnt/d/videos", lambda: f"{len(os.listdir('/mnt/d/videos'))} archivos"),
]

all_ok = all(check(label, fn) for label, fn in checks)

print()
if all_ok:
    print("Todo listo para correr el pipeline.")
else:
    print("Hay problemas que resolver antes de continuar.")
print()
