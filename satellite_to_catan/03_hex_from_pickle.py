"""
Hex Processor for Preprocessed Sentinel Pickle
----------------------------------------------
- Loads sentinel_256_final.pkl
- Applies hex mask
- Minimal edge fuzz
- Rejects low-variance tiles
- Outputs PNGs + pickle
"""

from pathlib import Path
import pickle
import numpy as np
import cv2

# =============================
# CONFIG
# =============================

HEX_SIZE = 256
EDGE_BLUR = 2
MIN_STD = 15
SAVE_PICKLE = True

# =============================
# PATHS (RELATIVE)
# =============================

BASE_DIR = Path(__file__).resolve().parent
IN_PKL = BASE_DIR / "data" / "sentinel_256_final.pkl"
OUT_DIR = BASE_DIR / "data" / "pkl_terrain_hexes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================
# HELPERS
# =============================

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


def is_valid(img):
    return img.std() >= MIN_STD


# =============================
# MAIN
# =============================

def main():
    print("📦 Loading pickle...")
    with open(IN_PKL, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        images = list(data.values())
    else:
        images = data

    print(f"🧩 Loaded {len(images)} tiles")

    mask = hex_mask(HEX_SIZE)
    all_hexes = []
    kept = 0

    for i, img in enumerate(images):
        if img.shape != (HEX_SIZE, HEX_SIZE, 3):
            continue

        if not is_valid(img):
            continue

        hex_img = (img * mask[..., None]).astype(np.uint8)

        fname = f"hex_{kept:06d}.png"
        cv2.imwrite(
            str(OUT_DIR / fname),
            cv2.cvtColor(hex_img, cv2.COLOR_RGB2BGR)
        )

        all_hexes.append(hex_img)
        kept += 1

    if SAVE_PICKLE:
        out_pkl = OUT_DIR / "terrain_hexes.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(all_hexes, f)
        print(f"💾 Saved pickle: {out_pkl}")

    print(f"\n✅ Done")
    print(f"Kept {kept} / {len(images)} tiles")
    print(f"Output dir: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
