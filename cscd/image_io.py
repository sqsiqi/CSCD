"""Small image I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Load an image as an RGB uint8 array."""
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image)


def save_rgb(image: np.ndarray, path: str | Path) -> None:
    """Save an RGB uint8 array."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(path)


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    """Save a binary mask as an 8-bit grayscale image."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)
