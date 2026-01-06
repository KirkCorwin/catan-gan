# ==========================
# Standard Catan Board Generator (minimal fuzz)
# ==========================

import sys
import subprocess
import importlib.util
import math
import random
from pathlib import Path

# ---------- AUTO-ENSURE DEPENDENCIES ----------
def ensure(pkg, import_name=None):
    import_name = import_name or pkg
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

ensure("numpy")
ensure("opencv-python", "cv2")
# ---------------------------------------------

import cv2
import numpy as np

# ---------- PATHS RELATIVE TO SCRIPT ----------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

INPUT_DIR = BASE_DIR / "hex_dataset/images"
OUTPUT_DIR = BASE_DIR / "catan_boards"
OUTPUT_DIR.mkdir(exist_ok=True)
# ----------------------------------------------

# -------- CONFIG ----------
HEX_RADIUS = 64        # size of each hex
FUZZY_BORDER = 2       # minimal blending
# Standard Catan row lengths
ROW_LENGTHS = [3, 4, 5, 4, 3]
# --------------------------

# Convert axial coords to pixel for flat-top hexes
def axial_to_pixel(q, r, size):
    x = size * 3/2 * q
    y = size * math.sqrt(3) * (r + q/2)
    return int(x), int(y)

# Load hex tiles
hex_files = list(INPUT_DIR.glob("*.png"))
if not hex_files:
    raise FileNotFoundError(f"No hex tiles found in {INPUT_DIR}!")

hex_images = [cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in hex_files]

# Estimate board size
board_width = int(HEX_RADIUS * 3/2 * 7 * 2)
board_height = int(math.sqrt(3) * HEX_RADIUS * 7 * 2)
board_img = np.zeros((board_height, board_width, 3), dtype=np.uint8)

center_x = board_width // 2
center_y = board_height // 2

# Generate axial coordinates
hex_coords = []
for r_idx, length in enumerate(ROW_LENGTHS):
    r_axial = r_idx - 2  # center row = 0
    q_start = -length // 2
    for q_idx in range(length):
        q_axial = q_start + q_idx
        hex_coords.append((q_axial, r_axial))

# Randomly select tiles
selected_tiles = random.choices(hex_images, k=len(hex_coords))

# Paste tiles with minimal fuzz
for (q, r), tile in zip(hex_coords, selected_tiles):
    dx, dy = axial_to_pixel(q, r, HEX_RADIUS)
    cx = center_x + dx
    cy = center_y + dy

    h, w = tile.shape[:2]
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = x0 + w, y0 + h

    if x0 < 0 or y0 < 0 or x1 > board_img.shape[1] or y1 > board_img.shape[0]:
        continue

    # Minimal fuzzy mask
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (w//2, h//2), HEX_RADIUS - FUZZY_BORDER, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (FUZZY_BORDER*2+1, FUZZY_BORDER*2+1), 0)

    roi = board_img[y0:y1, x0:x1]
    for c in range(3):
        roi[..., c] = roi[..., c] * (1 - mask) + tile[..., c] * mask

# Export board
output_file = OUTPUT_DIR / "catan_board.png"
cv2.imwrite(str(output_file), board_img)
print(f"✓ Standard Catan board generated and saved to: {output_file.resolve()}")
