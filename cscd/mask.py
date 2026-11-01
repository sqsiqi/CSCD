"""Mask utilities for data preparation and pixel composition."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def load_mask(path: str, size: Tuple[int, int] | None = None) -> np.ndarray:
    """Load a binary mask as a uint8 array with values in {0, 1}."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if size is not None:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Return the bounding box (x1, y1, x2, y2) of a binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def transfer_mask(
    ref_mask: np.ndarray,
    background_mask: np.ndarray,
    dilate: int = 0,
) -> np.ndarray:
    """Transfer the reference mask to the placement region.

    The reference mask is resized to the bounding box of ``background_mask``
    and pasted onto the target canvas. This follows the data protocol used in
    the paper; the core method only consumes the resulting mask.
    """
    h, w = background_mask.shape[:2]
    x1, y1, x2, y2 = bbox_from_mask(background_mask)
    target_w, target_h = max(1, x2 - x1), max(1, y2 - y1)

    resized = cv2.resize(ref_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((h, w), dtype=np.uint8)
    canvas[y1:y2, x1:x2] = resized
    canvas = (canvas & background_mask).astype(np.uint8)

    if dilate > 0:
        kernel = np.ones((dilate, dilate), dtype=np.uint8)
        canvas = cv2.dilate(canvas, kernel, iterations=1)
    return canvas


def compose_blend(
    background: np.ndarray,
    reference: np.ndarray,
    reference_mask: np.ndarray,
    background_mask: np.ndarray,
) -> np.ndarray:
    """Pixel-level composition used to initialize the diffusion process."""
    h, w = background.shape[:2]
    x1, y1, x2, y2 = bbox_from_mask(background_mask)
    target_w, target_h = max(1, x2 - x1), max(1, y2 - y1)

    ref = cv2.resize(reference, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    ref_mask = cv2.resize(reference_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    canvas = background.copy()
    roi = canvas[y1:y2, x1:x2]
    m = ref_mask[..., None].astype(np.float32)
    roi[:] = (m * ref + (1.0 - m) * roi).astype(np.uint8)
    return canvas
