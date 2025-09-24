Hybrid Keyframe Extraction Module
=================================

Purpose:
Two-stage hybrid algorithm combining lightweight motion/entropy heuristics with refinement to select salient keyframes efficiently.

Principles:
1. Stage 1 (Fast Pass): Low-cost metrics (e.g., frame diff, entropy) to shortlist candidate frames.
2. Stage 2 (Refinement): Applies more discriminative filters/morphology/LBP to finalize keyframes and reduce redundancy.
3. Config-Driven: All thresholds and tuning parameters externalized in YAML.
4. Extensible: Additional refinement heuristics can be appended without altering stage interfaces.

Advantages:
- Balances speed and quality.
- Avoids early heavy model inference.
- Provides clear tuning knobs for performance vs. recall.

Result: Produces structured keyframe outputs for downstream semantic indexing.