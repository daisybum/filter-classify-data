#!/usr/bin/env python3
"""Weather folder classifier using Swin Transformer V2.

This module exposes `classify_folder`, which predicts a weather class (0‒6)
for a directory of JPEG images, and `CLASS_NAMES`, the corresponding label
list. It is separated into its own file so that inference logic is cleanly
encapsulated and reusable.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from timm import create_model
from transformers import AutoModelForImageClassification

CLASS_NAMES: List[str] = [
    "Hazy",
    "Normal",
    "raining",
    "rainy but not raining",
    "snowing",
    "snowy but not snowing",
    "unclear",
]

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_MODEL: torch.nn.Module | None = None


def _load_model() -> torch.nn.Module:
    """Load Swin V2 classification model checkpoint lazily (singleton)."""
    global _MODEL  # noqa: PLW0603 – intentional module-level cache

    if _MODEL is not None:
        return _MODEL

    # Project root (= two levels up from this file) → weather_classification/models/
    ckpt_dir = Path(__file__).resolve().parents[2] / "weather_classification" / "models"
    ckpt_list = sorted(ckpt_dir.glob("*.pth"))
    if not ckpt_list:
        raise FileNotFoundError(
            f"No .pth checkpoint found in {ckpt_dir}. Please place your trained model there."  # noqa: E501
        )
    ckpt_path = ckpt_list[0]

    num_classes = len(CLASS_NAMES)

    ckpt = torch.load(ckpt_path, map_location=_DEVICE)
    state_dict = (
        ckpt.get("model_state_dict")
        or ckpt.get("state_dict")
        or ckpt  # raw state-dict fallback
    )

    try:
        model = create_model("swinv2_large", pretrained=False, num_classes=num_classes)
    except RuntimeError:
        model = AutoModelForImageClassification.from_pretrained(
            "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    model.load_state_dict(state_dict, strict=False)
    model.eval().to(_DEVICE)
    _MODEL = model
    return _MODEL


def classify_folder(folder: Path) -> int:
    """Predict weather class index for a folder of images.

    Parameters
    ----------
    folder : Path
        Directory containing JPEG images.

    Returns
    -------
    int
        Index 0‒6 corresponding to CLASS_NAMES.
    """
    model = _load_model()

    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]
    if not images:
        return CLASS_NAMES.index("unclear")

    probs_sum = torch.zeros(len(CLASS_NAMES), device=_DEVICE)

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        tensor = _PREPROCESS(img).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            output = model(tensor)
            logits = output.logits if hasattr(output, "logits") else output
            probs = F.softmax(logits, dim=1)[0]
            probs_sum += probs

    avg_probs = probs_sum / len(images)
    return int(avg_probs.argmax().item())
