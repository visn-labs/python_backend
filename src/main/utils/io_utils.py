"""I/O utilities for saving frames and masks."""
from __future__ import annotations
import os
import cv2
import numpy as np
from typing import Optional

__all__ = ["ensure_dir", "save_keyframe"]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_keyframe(frame, cfg, timestamp: float, stage: str, mask: Optional[np.ndarray] = None) -> bool:
    ts_str = f"{timestamp:.2f}".replace('.', '_')
    ensure_dir(cfg['output_dir'])
    out_name = f"kf_{ts_str}_{stage}.{cfg['image_extension']}"
    out_path = os.path.join(cfg['output_dir'], out_name)

    params = []
    if cfg['image_extension'].lower() in ("jpg", "jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(cfg.get('jpeg_quality', 90))]
    elif cfg['image_extension'].lower() == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    ok = cv2.imwrite(out_path, frame, params)
    if not ok:
        return False

    if cfg.get('save_debug_masks') and mask is not None:
        ensure_dir(cfg.get('mask_output_dir', 'hybrid_keyframes/masks'))
        mask_name = f"mask_{ts_str}_{stage}.png"
        cv2.imwrite(os.path.join(cfg.get('mask_output_dir', 'hybrid_keyframes/masks'), mask_name), mask)
    return True
