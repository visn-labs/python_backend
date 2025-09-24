"""Morphological processing helpers."""
from __future__ import annotations
import cv2
import numpy as np

__all__ = ["apply_morph"]

def apply_morph(mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
