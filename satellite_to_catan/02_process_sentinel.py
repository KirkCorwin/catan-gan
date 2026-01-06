# ==========================
# 02_process_sentinel.py
# Terrain preprocessing for Catan-style maps
# ==========================

import sys
import subprocess
import importlib.util
from pathlib import Path

def ensure(pkg, import_name=None):
    import_name = import_name or pkg
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

ensure("opencv-python", "cv2")
ensure("numpy")

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
IN_DIR = BASE_DIR / "raw_images/sentinel"
OUT_DIR = BASE_DIR / "cleaned_images"
OUT_DIR.mkdir(exist_ok=True)

DOWNSCALE = 0.75
COLOR_STEP = 16
BLUR_KERNEL = 7

for img_path in IN_DIR.glob("*.png"):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    img = cv2.resize(img, None, fx=DOWNSCALE, fy=DOWNSCALE, interpolation=cv2.INTER_AREA)
    img = cv2.bilateralFilter(img, BLUR_KERNEL, 50, 50)
    img = (img // COLOR_STEP) * COLOR_STEP

    cv2.imwrite(str(OUT_DIR / img_path.name), img)
    print(f"✓ Processed {img_path.name}")

print("✓ Terrain preprocessing complete")
