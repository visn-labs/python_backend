"""Hybrid keyframe extraction pipeline orchestrating Stage 1 and Stage 2."""
from __future__ import annotations
import os
import time
import cv2
from typing import List, Dict, Any
from tqdm import tqdm

from src.main.utils.config_loader import load_configs
from src.main.utils.io_utils import ensure_dir, save_keyframe, flush_dir
from src.main.keyframe_extractor.stages.stage1 import Stage1Motion
from src.main.keyframe_extractor.stages.stage2 import Stage2Texture
from src.main.keyframe_extractor.adaptive_threshold import AdaptiveThresholdManager

__all__ = ["HybridPipeline", "load_hybrid_settings"]


def load_hybrid_settings(config_dir: str) -> Dict[str, Any]:
    base = os.path.join(config_dir, 'base.yml')
    stage1 = os.path.join(config_dir, 'stage1.yml')
    stage2 = os.path.join(config_dir, 'stage2.yml')
    debug = os.path.join(config_dir, 'debug.yml')
    merged = load_configs([base, stage1, stage2, debug])
    return merged


class HybridPipeline:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.stage1 = Stage1Motion(cfg)
        self.stage2 = Stage2Texture(cfg)
        # Adaptive threshold manager initialization
        # Support both nested (legacy) and flattened adaptive config.
        adapt_cfg_nested = cfg.get('adaptive_threshold') or {}
        def _cfg(name, default):
            return cfg.get(f'adaptive_{name}', adapt_cfg_nested.get(name, default))
        self.adaptive_enabled = _cfg('enabled', True)
        self.adaptive = AdaptiveThresholdManager(
            window_size=_cfg('window_size', 300),
            k_low=_cfg('k_low', 1.0),
            k_high=_cfg('k_high', 3.0),
            min_history=_cfg('min_history', 30),
            default_low=cfg.get('low_motion_threshold', 5000),
            default_high=cfg.get('high_motion_threshold', 10000),
            smooth_factor=_cfg('smooth_factor', 0.0),
        )

    def run(self) -> List[float]:
        cfg = self.cfg
        video_path = cfg['video_path']
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Fresh run: clear previous contents to avoid mixing outputs from different videos
        flush_dir(cfg['output_dir'])
        if cfg.get('save_debug_masks'):
            flush_dir(cfg.get('mask_output_dir', 'hybrid_keyframes/masks'))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        min_distance_frames = max(1, int(cfg.get('min_distance_seconds', 1.0) * fps))

        frame_index = 0
        last_saved_frame = -min_distance_frames
        keyframe_timestamps: List[float] = []
        saved_count = 0

        iterator = range(total_frames) if total_frames > 0 else iter(int, 1)
        progress_iter = tqdm(iterator, total=total_frames, disable=not cfg.get('progress_bar', True), desc="Hybrid Pipeline")

        start_time = time.time()

        for _ in progress_iter:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            if cfg.get('frame_step', 1) > 1 and (frame_index % cfg['frame_step']) != 0:
                continue

            if cfg.get('resize_scale', 1.0) != 1.0:
                frame = cv2.resize(frame, None, fx=cfg['resize_scale'], fy=cfg['resize_scale'], interpolation=cv2.INTER_AREA)

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if cfg.get('force_grayscale', True) else frame

            # Stage 1 motion & adaptive thresholds
            motion1, mask1 = self.stage1.process(frame_gray)
            if self.adaptive_enabled:
                low_thr, high_thr = self.adaptive.update(motion1)
            else:
                low_thr = cfg.get('low_motion_threshold', 5000)
                high_thr = cfg.get('high_motion_threshold', 10000)

            if motion1 <= low_thr:
                continue

            timestamp = frame_index / fps

            # Immediate accept
            if motion1 >= high_thr:
                if frame_index - last_saved_frame >= min_distance_frames:
                    if save_keyframe(frame, cfg, timestamp, 'color', mask1 if cfg.get('save_debug_masks') else None):
                        keyframe_timestamps.append(timestamp)
                        last_saved_frame = frame_index
                        saved_count += 1
                        if cfg.get('max_keyframes') and saved_count >= cfg['max_keyframes']:
                            break
                continue

            # Stage 2
            motion2, mask2, _ = self.stage2.process(frame_gray)
            # Recompute thresholds after motion2? Keep same for consistency within frame.
            if motion2 >= high_thr and frame_index - last_saved_frame >= min_distance_frames:
                if save_keyframe(frame, cfg, timestamp, 'texture', mask2 if cfg.get('save_debug_masks') else None):
                    keyframe_timestamps.append(timestamp)
                    last_saved_frame = frame_index
                    saved_count += 1
                    if cfg.get('max_keyframes') and saved_count >= cfg['max_keyframes']:
                        break

        cap.release()
        elapsed = time.time() - start_time
        if cfg.get('verbose', True):
            print(f"Hybrid pipeline finished. {len(keyframe_timestamps)} keyframes saved in {elapsed:.2f}s")
        return keyframe_timestamps


def main():
    root = os.path.dirname(__file__)
    config_dir = os.path.join(root, 'config')
    cfg = load_hybrid_settings(config_dir)
    pipeline = HybridPipeline(cfg)
    pipeline.run()


if __name__ == '__main__':
    main()
