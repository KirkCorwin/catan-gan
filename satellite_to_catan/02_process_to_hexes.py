"""
Sentinel-2 Scene → Filtered Terrain Hexes
----------------------------------------
- Full-scene hex grid extraction
- SCL-based rejection (>=75% bad pixels)
- Gamma correction
- Low-variance rejection
- Minimal edge fuzz
- GAN-ready output
"""

from pathlib import Path
import numpy as np
import rasterio
import cv2
import pickle
import math

# =============================
# CONFIG
# =============================

HEX_SIZE = 256

HEX_H_SPACING = 0.75
HEX_V_SPACING = math.sqrt(3) / 2

EDGE_BLUR = 2
SAVE_PICKLE = True

# Reject if >= 75% of pixels are undesirable
BAD_PIXEL_THRESHOLD = 0.75

# SCL values to reject
BAD_SCL_VALUES = {3, 6, 8, 9, 10, 11}

# Reject very flat tiles (water, ice, desert)
MIN_STD = 15

# =============================
# PATHS
# =============================

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw_sentinel"
OUT_DIR = BASE_DIR / "data" / "terrain_hexes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================
# HELPERS
# =============================

def load_band(scene_dir, name):
    with rasterio.open(scene_dir / f"{name}.jp2") as src:
        return src.read(1)


def load_scene(scene_dir):
    """Load RGB + SCL"""
    R = load_band(scene_dir, "B04")
    G = load_band(scene_dir, "B03")
    B = load_band(scene_dir, "B02")
    SCL = load_band(scene_dir, "SCL")

    rgb = np.stack([R, G, B], axis=-1).astype(np.float32)

    # Robust normalization
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)

    # Gamma correction (critical)
    rgb = rgb ** (1 / 2.2)

    rgb = (rgb * 255).astype(np.uint8)
    return rgb, SCL


def hex_mask(size):
    mask = np.zeros((size, size), dtype=np.uint8)
    r = size // 2
    cx = cy = r

    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6
    pts = np.stack([
        cx + r * np.cos(angles),
        cy + r * np.sin(angles)
    ], axis=1).astype(np.int32)

    cv2.fillConvexPoly(mask, pts, 255)

    if EDGE_BLUR > 0:
        mask = cv2.GaussianBlur(mask, (EDGE_BLUR * 2 + 1,) * 2, 0)

    return mask.astype(np.float32) / 255.0


def is_valid_hex(scl_patch):
    bad = np.isin(scl_patch, list(BAD_SCL_VALUES))
    return bad.mean() < BAD_PIXEL_THRESHOLD


def extract_hex(img, scl, cx, cy, r, mask):
    x0, y0 = int(cx - r), int(cy - r)
    x1, y1 = int(cx + r), int(cy + r)

    rgb_patch = img[y0:y1, x0:x1]
    scl_patch = scl[y0:y1, x0:x1]

    if rgb_patch.shape[:2] != (HEX_SIZE, HEX_SIZE):
        return None

    if not is_valid_hex(scl_patch):
        return None

    if rgb_patch.std() < MIN_STD:
        return None

    return (rgb_patch * mask[..., None]).astype(np.uint8)

# =============================
# MAIN
# =============================

def main():
    mask = hex_mask(HEX_SIZE)
    r = HEX_SIZE // 2
    step_x = int(HEX_SIZE * HEX_H_SPACING)
    step_y = int(HEX_SIZE * HEX_V_SPACING)

    all_hexes = []
    hex_id = 0

    scenes = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    print(f"🔍 Found {len(scenes)} scenes")

    for scene in scenes:
        print(f"\n🗺 Processing {scene.name}")
        img, scl = load_scene(scene)
        h, w, _ = img.shape

        row = 0
        for y in range(r, h - r, step_y):
            offset = 0 if row % 2 == 0 else step_x // 2

            for x in range(r + offset, w - r, step_x):
                hex_img = extract_hex(img, scl, x, y, r, mask)
                if hex_img is None:
                    continue

                fname = f"{scene.name}_hex_{hex_id:06d}.png"
                cv2.imwrite(
                    str(OUT_DIR / fname),
                    cv2.cvtColor(hex_img, cv2.COLOR_RGB2BGR)
                )

                all_hexes.append(hex_img)
                hex_id += 1

            row += 1

        print(f"  → Total hexes so far: {hex_id}")

    if SAVE_PICKLE:
        pkl_path = OUT_DIR / "terrain_hexes.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(all_hexes, f)
        print(f"\n💾 Pickle saved: {pkl_path}")

    print("\n🎉 Terrain hex extraction complete")
    print(f"Final hex count: {len(all_hexes)}")
    print(f"Output directory: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
