"""Stage 2: Texture-based confirmation using LBP + MOG2."""
from __future__ import annotations
import cv2
from src.main.utils.lbp import compute_lbp
from src.main.utils.morph import apply_morph

__all__ = ["Stage2Texture"]

class Stage2Texture:
    def __init__(self, cfg: dict):
        mog_cfg = cfg.get('mog_texture', {})
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=mog_cfg.get('history', 500),
            varThreshold=mog_cfg.get('varThreshold', 16.0),
            detectShadows=mog_cfg.get('detectShadows', True)
        )
        lbp_cfg = cfg.get('lbp', {})
        self.radius = lbp_cfg.get('radius', 1)
        self.points = lbp_cfg.get('points', 8)
        self.high = cfg.get('high_motion_threshold', 5000)
        self.apply_morphology = cfg.get('apply_morphology', True)
        self.morph_kernel_size = cfg.get('morph_kernel_size', 3)

    def process(self, frame_gray):
        lbp_img = compute_lbp(frame_gray, self.radius, self.points)
        mask = self.subtractor.apply(lbp_img)
        if self.apply_morphology:
            mask = apply_morph(mask, self.morph_kernel_size)
        motion = int(cv2.countNonZero(mask))
        return motion, mask, lbp_img
