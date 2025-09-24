"""Reasoning (VLM) filter.
Delegates visual yes/no verification to utils.vlm.vlm_yes_no (Gemini / stub).
"""
from __future__ import annotations
from typing import Any
from ..utils.vlm import vlm_yes_no
import os


def reasoning_filter(candidates, question: str, enable: bool, max_items: int | None,
                     provider: str, model: str, api_key_env: str, keyframes_dir: str | None):
    """Return confirmed candidates plus audit trail.

    Returns (confirmed, audit) where audit is list of dicts.
    """
    # Behavior overview (post-fix):
    # - Indexer now stores absolute disk path under metadata['abs_path'] plus a run_id.
    # - We prefer that path; if absent, we fall back to joining keyframes_dir + record.id.
    # - If the image file no longer exists (stale vector store entry), we skip it and
    #   record an audit entry with decision 'Skipped (missing file)' so downstream logic
    #   never queries a VLM on non-existent frames.
    confirmed = []
    audit = []
    for c in candidates:
        decision = True
        if enable:
            # Prefer absolute path from metadata (new field); fallback to join logic.
            meta_path = c.metadata.get('abs_path')
            img_path = None
            if isinstance(meta_path, str) and os.path.isabs(meta_path):
                img_path = meta_path
            else:
                tmp = c.id
                if keyframes_dir and not os.path.isabs(tmp):
                    tmp_full = os.path.join(keyframes_dir, tmp)
                    if os.path.exists(tmp_full):
                        img_path = tmp_full
                if img_path is None and os.path.exists(tmp):
                    img_path = tmp
            if not img_path or not os.path.exists(img_path):
                audit.append({
                    'frame_id': c.id,
                    'question': question,
                    'decision': 'Skipped (missing file)',
                    'timestamp': c.metadata.get('video_timestamp')
                })
                continue
            decision = vlm_yes_no(img_path, question, provider=provider, model=model, api_key_env=api_key_env)
        audit.append({
            'frame_id': c.id,
            'question': question,
            'decision': 'Yes' if decision else 'No',
            'timestamp': c.metadata.get('video_timestamp')
        })
        if decision:
            confirmed.append(c)
        if max_items and len(confirmed) >= max_items:
            break
    return confirmed, audit

__all__ = ["reasoning_filter"]
