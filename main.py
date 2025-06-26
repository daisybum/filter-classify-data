#!/usr/bin/env python3
"""
filter_and_copy.py

✓ Scan date/time named folders inside source root
✓ If there's at least one JPEG/JPG file, classify → copy
✓ Label 0-6 → ['Hazy', 'Normal', 'raining', 'rainy but not raining',
               'snowing', 'snowy but not snowing', 'unclear']
"""

import argparse
import json
import shutil
from pathlib import Path

# Third-party ML & CV

from src.classifier import CLASS_NAMES, classify_folder
from tqdm import tqdm


# ────────────────────────────── Utility Functions ──────────────────────────────
def has_jpeg(folder: Path) -> bool:
    """Returns True if folder contains at least one .jpg/.jpeg file."""
    return any(p.suffix.lower() in {'.jpg', '.jpeg'} for p in folder.iterdir() if p.is_file())



# ────────────────────────────── Main Logic ──────────────────────────────
def main(src_root: Path, dst_root: Path, overwrite: bool):
    # Pre-create 7 class folders at destination
    for name in CLASS_NAMES:
        (dst_root / name).mkdir(parents=True, exist_ok=True)

    # Step 1: Iterate through 1-depth subfolders of source root
    for subdir in tqdm([p for p in src_root.iterdir() if p.is_dir()], desc="Folders", unit="dir"):
        if not has_jpeg(subdir):            # Skip folders without JPEG
            continue

        label = classify_folder(subdir)     # Predict 0-6 label
        dst_class_dir = dst_root / CLASS_NAMES[label]
        dst_path = dst_class_dir / subdir.name

        # Step 2: Copy folder (including metadata, Python≥3.8)
        if dst_path.exists() and not overwrite:
            print(f'‼️ Already exists → Skip: {dst_path}')
            continue
        shutil.copytree(subdir, dst_path, dirs_exist_ok=overwrite)
        print(f'✅ {subdir}  →  {dst_path}')

# ────────────────────────────── CLI Entry Point ──────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JPEG-containing folder filtering, classification, and copying script")
    parser.add_argument("--config", default="config.json", help="Path to JSON config file containing src/dst/overwrite")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as cf:
        cfg = json.load(cf)

    main(
        Path(cfg["src"]).expanduser(),
        Path(cfg["dst"]).expanduser(),
        bool(cfg.get("overwrite", False)),
    )