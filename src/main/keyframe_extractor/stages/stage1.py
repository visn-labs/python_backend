"""Stage 1: Fast MOG2 motion detection."""
from __future__ import annotations
import cv2
from src.main.utils.morph import apply_morph

__all__ = ["Stage1Motion"]

class Stage1Motion:
    def __init__(self, cfg: dict):
        mog_cfg = cfg.get('mog_color', {})
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=mog_cfg.get('history', 500),
            varThreshold=mog_cfg.get('varThreshold', 16.0),
            detectShadows=mog_cfg.get('detectShadows', True)
        )
        self.low = cfg.get('low_motion_threshold', 200)
        self.high = cfg.get('high_motion_threshold', 5000)
        self.apply_morphology = cfg.get('apply_morphology', True)
        self.morph_kernel_size = cfg.get('morph_kernel_size', 3)

    def process(self, frame_gray):
        mask = self.subtractor.apply(frame_gray)
        if self.apply_morphology:
            mask = apply_morph(mask, self.morph_kernel_size)
        motion = int(cv2.countNonZero(mask))
        return motion, mask
