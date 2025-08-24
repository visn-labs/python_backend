"""Local Binary Pattern (LBP) utilities."""
from __future__ import annotations
import numpy as np
import math

__all__ = ["compute_lbp"]

def compute_lbp(gray: np.ndarray, radius: int, points: int) -> np.ndarray:
    """Compute a basic LBP image (NumPy)."""
    padded = np.pad(gray, radius, mode='edge')
    h, w = gray.shape
    lbp = np.zeros_like(gray, dtype=np.uint8)
    angles = np.linspace(0, 2 * math.pi, points, endpoint=False)
    for idx, angle in enumerate(angles):
        dy = int(round(radius * math.sin(angle)))
        dx = int(round(radius * math.cos(angle)))
        shifted = padded[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
        lbp |= ((shifted >= gray) << idx).astype(np.uint8)
    return lbp
