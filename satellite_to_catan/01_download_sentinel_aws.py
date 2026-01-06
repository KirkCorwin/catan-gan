"""
Sentinel-2 Global Terrain Sampler (Curated, Robust)
--------------------------------------------------
- Downloads RGB + SCL bands
- Uses unsigned AWS access (free)
- Selects 15 geographically diverse tiles:
    * 1 Arctic ocean
    * 1 Tropical ocean
    * 13 varied land environments
- Auto-discovers valid scenes (no 404s)
- Paths are relative to this script
"""

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path
import random

# =============================
# CONFIG
# =============================

# Required bands for processing
BANDS = ["B02", "B03", "B04", "SCL"]

# One scene per tile (enough for diversity)
SCENES_PER_TILE = 1

# =============================
# TILE SELECTION (CURATED)
# =============================

# Ocean tiles
ARCTIC_OCEAN_TILE = "33/X/VP"      # Arctic Ocean
TROPICAL_OCEAN_TILE = "31/N/EA"    # Tropical Atlantic

# Land tiles (diverse climates & continents)
LAND_TILES = [
    "32/T/QR",   # Central Europe (Germany)
    "10/S/EG",   # California
    "18/N/UL",   # Andes
    "22/M/DB",   # Amazon
    "36/R/UU",   # East Africa
    "54/H/VF",   # Australia
    "48/P/VS",   # Southeast Asia
    "45/R/VL",   # Himalayas
    "29/R/NK",   # West Africa
    "38/S/LC",   # Middle East
    "20/H/PH",   # Canadian Shield
    "56/J/KL",   # New Zealand
    "43/Q/EA",   # Central Asia
]

# Final tile list (15 total)
TILES = [ARCTIC_OCEAN_TILE, TROPICAL_OCEAN_TILE] + LAND_TILES

assert len(TILES) == 15, "Tile count must be exactly 15"

# =============================
# PATHS (RELATIVE)
# =============================

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw_sentinel"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# =============================
# AWS S3 CLIENT (UNSIGNED)
# =============================

BUCKET = "sentinel-s2-l2a"

s3 = boto3.client(
    "s3",
    config=Config(signature_version=UNSIGNED)
)

# =============================
# HELPERS
# =============================

def list_prefixes(prefix):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=prefix,
        Delimiter="/"
    ):
        for cp in page.get("CommonPrefixes", []):
            yield cp["Prefix"]


def discover_scenes(tile):
    """Return all available scene prefixes for a tile"""
    scenes = []
    base = f"tiles/{tile}/"

    for year in list_prefixes(base):
        for month in list_prefixes(year):
            for day in list_prefixes(month):
                for seq in list_prefixes(day):
                    scenes.append(seq)

    return scenes


def download_scene(scene_prefix, out_dir):
    """Download required bands for one scene"""
    out_dir.mkdir(parents=True, exist_ok=True)

    for band in BANDS:
        key = f"{scene_prefix}R10m/{band}.jp2" if band != "SCL" else f"{scene_prefix}R20m/SCL.jp2"
        local = out_dir / f"{band}.jp2"

        print(f"    ↓ {band}")
        s3.download_file(BUCKET, key, local)


# =============================
# MAIN
# =============================

def main():
    print("🌍 Sentinel-2 Global Terrain Sampling")
    print(f"Downloading {len(TILES)} tiles\n")

    for tile in TILES:
        print(f"🔎 Tile {tile}")
        scenes = discover_scenes(tile)

        if not scenes:
            print("  ⚠ No scenes found, skipping")
            continue

        selected = random.sample(scenes, min(SCENES_PER_TILE, len(scenes)))

        for i, scene in enumerate(selected):
            scene_id = scene.rstrip("/").split("/")[-1]
            out_dir = RAW_DIR / f"{tile.replace('/', '_')}_scene_{i}_{scene_id}"

            print(f"  ⬇ Scene {i+1}: {scene}")
            download_scene(scene, out_dir)

    print("\n🎉 Download complete")
    print(f"Data saved to: {RAW_DIR.resolve()}")


if __name__ == "__main__":
    main()
