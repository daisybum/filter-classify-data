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


# ────────────────────────────── Utility Functions ──────────────────────────────
def has_jpeg(folder: Path) -> bool:
    """Returns True if folder contains at least one .jpg/.jpeg file."""
    return any(p.suffix.lower() in {'.jpg', '.jpeg'} for p in folder.iterdir() if p.is_file())



"""
    Strategy
    --------
    1. Lazily load the Swin-Transformer V2 model (checkpoint searched under
       ``weather_classification/models/*.pth``).
    2. Run inference on every JPEG in *folder*.
    3. Average the softmax probabilities across images and return the arg-max
       class index (0-6).
    """

    global _MODEL  # noqa: PLW0603 – mutation intentional for caching

    # ── Lazy model init ────────────────────────────────────────────────────────
    if _MODEL is None:
        ckpt_dir = (Path(__file__).resolve().parent.parent / "weather_classification" / "models")
        ckpt_path_list = sorted(ckpt_dir.glob("*.pth"))
        if not ckpt_path_list:
            raise FileNotFoundError(
                f"No .pth checkpoint found in {ckpt_dir}. Please place your trained model there.")
        ckpt_path = ckpt_path_list[0]  # pick the first one

        # Always 7 classes for this project
        num_classes = len(CLASS_NAMES)
        device = _DEVICE

        # ----------------------- Checkpoint loading --------------------------
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = (
            ckpt.get("model_state_dict")
            or ckpt.get("state_dict")
            or ckpt  # assume raw state-dict
        )

        # ---------------------------- Model ----------------------------------
        try:
            model = create_model("swinv2_large", pretrained=False, num_classes=num_classes)
        except RuntimeError:
            model = AutoModelForImageClassification.from_pretrained(
                "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )

        model.load_state_dict(state_dict, strict=False)
        model.eval().to(device)
        _MODEL = model  # cache

    model = _MODEL

    # ── Image preprocessing ───────────────────────────────────────────────────
    preprocess = _PREPROCESS

    # Iterate over JPEG images in the folder
    img_paths = [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}]
    if not img_paths:
        return CLASS_NAMES.index("unclear")  # default fallback

    probs_accum = torch.zeros(len(CLASS_NAMES), device=_DEVICE)
    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            output = model(tensor)
            logits = output.logits if hasattr(output, "logits") else output
            probs = F.softmax(logits, dim=1)[0]
            probs_accum += probs

    avg_probs = probs_accum / len(img_paths)
    predicted_idx = int(avg_probs.argmax().item())
"""

# ────────────────────────────── Main Logic ──────────────────────────────
def main(src_root: Path, dst_root: Path, overwrite: bool):
    # Pre-create 7 class folders at destination
    for name in CLASS_NAMES:
        (dst_root / name).mkdir(parents=True, exist_ok=True)

    # Step 1: Iterate through 1-depth subfolders of source root
    for subdir in [p for p in src_root.iterdir() if p.is_dir()]:
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