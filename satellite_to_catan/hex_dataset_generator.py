# === Automatic Hex-Tile Dataset Generator (Catan-style, flat-top) ===
# Copy-paste this entire cell and run

# # --- auto-install dependencies if missing ---
# import sys
# import subprocess

# def ensure(pkg):
#     try:
#         __import__(pkg)
#     except ImportError:
#         subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# for p in ["numpy", "pandas", "opencv-python", "pyarrow"]:
#     ensure(p)
# # -------------------------------------------



import math
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent



# ---------------- CONFIG ----------------
INPUT_DIR  = BASE_DIR / "input_images"
OUTPUT_DIR = BASE_DIR / "hex_dataset"

HEX_RADIUS = 64                 # pixels (center -> corner)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
# ----------------------------------------


def axial_to_pixel(q, r, size):
    x = size * (3 / 2 * q)
    y = size * (math.sqrt(3) * (r + q / 2))
    return x, y


def hex_grid_for_image(width, height, size):
    cols = int(width / (1.5 * size)) + 3
    rows = int(height / (math.sqrt(3) * size)) + 3
    for q in range(-cols, cols + 1):
        for r in range(-rows, rows + 1):
            yield q, r


def hex_mask(cx, cy, r, shape):
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    pts = np.stack([
        cx + r * np.cos(angles),
        cy + r * np.sin(angles)
    ], axis=1).astype(np.int32)

    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def extract_hex(image, cx, cy, r):
    h, w = image.shape[:2]
    x0, y0 = int(cx - r), int(cy - r)
    x1, y1 = int(cx + r), int(cy + r)

    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None

    mask = hex_mask(cx, cy, r, image.shape)
    crop = image[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]

    crop[crop_mask == 0] = 0
    return crop


def build_hex_dataset(input_dir, output_dir, hex_radius):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    img_out = output_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    records = []

    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        cx_img, cy_img = w // 2, h // 2

        for q, r in hex_grid_for_image(w, h, hex_radius):
            dx, dy = axial_to_pixel(q, r, hex_radius)
            cx = int(cx_img + dx)
            cy = int(cy_img + dy)

            tile = extract_hex(img, cx, cy, hex_radius)
            if tile is None:
                continue

            fname = f"{uuid.uuid4().hex}.png"
            cv2.imwrite(str(img_out / fname), tile)

            records.append({
                "file": fname,
                "source_image": img_path.name,
                "q": q,
                "r": r,
                "x": cx,
                "y": cy
            })

    df = pd.DataFrame(records)
    df.to_parquet(output_dir / "metadata.parquet")
    print(f"✓ Generated {len(df)} hex tiles")
    print(f"✓ Dataset saved to: {output_dir.resolve()}")


# ---------------- RUN ----------------
build_hex_dataset(INPUT_DIR, OUTPUT_DIR, HEX_RADIUS)
