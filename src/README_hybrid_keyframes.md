# Hybrid Two-Stage Keyframe Extraction

This module implements an efficient, CPU-friendly, *texture-aware* keyframe extraction system combining:

1. **Stage 1 (Fast):** Motion via background subtraction (MOG2) on grayscale/color frames.
2. **Stage 2 (Precise):** Local Binary Pattern (LBP) texture map + separate MOG2 for borderline motion.

Designed to:
- Skip obvious non-motion frames quickly.
- Accept obvious high-motion frames immediately.
- Use texture analysis only where subtle motion might matter ("maybe zone").
- Handle camouflaged objects whose color blends into the background.

---
## Quick Start

1. Edit the configuration: `resources/hybrid_keyframes.yml`
2. (Optional) Downscale your video first using `downscaler.py` for speed.
3. Run the extractor:

```
python repo/hybrid_keyframes.py
```

Keyframes will be written to the configured `output_dir`.

---
## Configuration (YAML)

| Key | Description |
|-----|-------------|
| video_path | Path to the input video (ideally already downscaled). |
| output_dir | Directory for saved keyframes. |
| resize_scale | Additional internal scaling (e.g., 0.5 halves dimensions). |
| frame_step | Process every Nth frame. 2 = half workload. |
| min_distance_seconds | Minimum time spacing between saved keyframes. |
| low_motion_threshold | Pixel count (post MOG2) to enter the maybe zone. |
| high_motion_threshold | Pixel count to auto-accept as keyframe. |
| lbp.radius / lbp.points | LBP neighborhood. Points often = 8 * radius. |
| mog_color / mog_texture | Background subtractor parameters for each stage. |
| morph_kernel_size | Size of morphological kernel (odd). |
| apply_morphology | Enable noise cleanup with open+close. |
| save_debug_masks | Save binary masks for inspection. |
| mask_output_dir | Directory for saved masks. |
| image_extension | jpg or png. |
| jpeg_quality | JPEG quality (if jpg). |
| max_keyframes | 0 = unlimited, else stop after this many. |

See full defaults in `resources/hybrid_keyframes.yml`.

---
## How It Works

1. **Frame Acquisition & Downscaling** – Optionally downscale externally (recommended) and/or via `resize_scale` for speed.
2. **Stage 1 (Color/Gray MOG2)** – Apply MOG2 background subtraction. Count foreground pixels. If below `low_motion_threshold` => discard; if above `high_motion_threshold` => save keyframe.
3. **Stage 2 (Texture MOG2)** – For frames in the maybe zone, compute an LBP texture map. Run a separate MOG2 subtractor trained on these texture images. If foreground texture pixels exceed `high_motion_threshold`, save keyframe.
4. **Spacing Constraint** – Enforce `min_distance_seconds` to avoid near-duplicate frames.

---
## Performance Tuning

| Goal | Adjust |
|------|--------|
| Faster runtime | Increase `frame_step`, lower `resize_scale`, raise `low_motion_threshold`. |
| More sensitivity | Lower `low_motion_threshold` & `high_motion_threshold`. |
| Fewer duplicates | Raise `min_distance_seconds`. |
| Better subtle motion detection | Keep `low_motion_threshold` low but not too low (avoid noise); maybe increase `lbp.radius`. |
| Less noise | Enable `apply_morphology`, tweak `morph_kernel_size` (3 or 5). |

---
## Extensibility

Potential improvements:
- Multi-scale LBP variants.
- Replace pixel count with connected component area filtering.
- Add optical flow magnitude gating.
- GPU acceleration for heavy videos.

---
## Code Integration

```python
from hybrid_keyframes import load_hybrid_config, run_hybrid_extraction
cfg = load_hybrid_config('resources/hybrid_keyframes.yml')
keyframes = run_hybrid_extraction(cfg)
print(keyframes)
```

---
## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| 0 keyframes | Thresholds too high | Lower `low_motion_threshold` / `high_motion_threshold`. |
| Too many keyframes | Thresholds too low | Raise thresholds or increase `min_distance_seconds`. |
| Slow processing | Large frames or frame_step=1 | Downscale video / set `resize_scale` < 1 / increase `frame_step`. |
| Masks noisy | Fine texture or sensor noise | Increase `morph_kernel_size` or thresholds. |

---
## License

Internal usage blueprint implementation.
