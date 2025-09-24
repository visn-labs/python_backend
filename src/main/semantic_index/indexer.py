"""Offline indexing pipeline for keyframes.
Parses filenames for temporal metadata, computes embeddings, stores vector records.
"""
from __future__ import annotations
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass
import uuid
import numpy as np
from glob import glob
from .embedding import load_embedding_backends
from .vector_store import load_vector_store

ISO_FORMATS = [
    '%Y-%m-%dT%H:%M:%S',
    '%Y-%m-%d %H:%M:%S',
]

@dataclass
class IndexingConfig:
    keyframes_dir: str
    patterns: List[str]
    filename_seconds_regex: str
    video_start_time: str | None
    timezone: str | None
    vector_store: str
    vector_store_dir: str
    image_embedding_model: str
    text_embedding_model: str
    embed_batch_size: int
    embedding_dim: int | None
    extra_metadata: Dict[str, Any]
    skip_on_error: bool
    flush_interval: int
    log_level: str
    run_id: str | None


def parse_time(ts: str | None) -> datetime | None:
    if not ts:
        return None
    for fmt in ISO_FORMATS:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported datetime format: {ts}")


def build_indexing_config(raw: Dict[str, Any]) -> IndexingConfig:
    return IndexingConfig(
        keyframes_dir=raw['keyframes_dir'],
        patterns=raw.get('patterns', ['*.jpg']),
        filename_seconds_regex=raw.get('filename_seconds_regex', r'(?P<seconds>\\d+(?:\\.\\d+)?)s'),
        video_start_time=raw.get('video_start_time'),
        timezone=raw.get('timezone'),
        vector_store=raw.get('vector_store', 'inmem'),
        vector_store_dir=raw.get('vector_store_dir', '.vector_store'),
        image_embedding_model=raw.get('image_embedding_model', 'openai/clip-vit-base-patch32'),
        text_embedding_model=raw.get('text_embedding_model', 'openai/clip-vit-base-patch32'),
        embed_batch_size=int(raw.get('embed_batch_size', 16)),
        embedding_dim=raw.get('embedding_dim'),
        extra_metadata=raw.get('extra_metadata', {}),
        skip_on_error=bool(raw.get('skip_on_error', True)),
        flush_interval=int(raw.get('flush_interval', 200)),
    log_level=raw.get('log_level', 'INFO'),
    run_id=raw.get('run_id'),
    )


def discover_keyframes(cfg: IndexingConfig) -> List[str]:
    files: List[str] = []
    for pat in cfg.patterns:
        files.extend(glob(os.path.join(cfg.keyframes_dir, pat)))
    files.sort()
    return files


def extract_seconds(fname: str, regex: str) -> float | None:
    """Extract seconds from keyframe filename.

    Original config expected pattern like: frame_0045s.jpg (regex with trailing 's').
    Current keyframe saver emits names like: kf_12_34_color.jpg (timestamp with underscore,
    no trailing 's'). This caused zero matches and all frames skipped, producing an empty index.

    Strategy:
      1. Try provided regex first (backwards compatibility).
      2. Fallback patterns for current naming: kf_<sec>_<stage>.* capturing either
         underscore or decimal separator.
      3. Replace underscore with '.' before float conversion.
    """
    base = os.path.basename(fname)
    m = re.search(regex, base)
    if not m:
        fallback_patterns = [
            r"kf_(?P<seconds>\d+_\d+)_",         # e.g. kf_12_34_color.jpg
            r"kf_(?P<seconds>\d+(?:\.\d+)?)_",  # e.g. kf_12.34_color.jpg (future-proof)
        ]
        for pat in fallback_patterns:
            m = re.search(pat, base)
            if m:
                break
    if not m:
        return None
    raw = m.group('seconds').replace('_', '.')
    try:
        return float(raw)
    except Exception:
        return None


def index_keyframes(raw_cfg: Dict[str, Any]):
    cfg = build_indexing_config(raw_cfg)
    start_dt = parse_time(cfg.video_start_time)
    run_id = cfg.run_id or str(uuid.uuid4())

    img_backend, _, dim = load_embedding_backends(
        cfg.image_embedding_model, cfg.text_embedding_model, cfg.embedding_dim
    )
    # Ensure dim is plain int (open-clip path may yield tensor-like); fall back to config override
    dim_int = int(dim) if not isinstance(dim, int) else dim
    store = load_vector_store(cfg.vector_store, dim_int, cfg.vector_store_dir)

    keyframes = discover_keyframes(cfg)
    if cfg.log_level == 'INFO':
        print(f"Discovered {len(keyframes)} keyframes in {cfg.keyframes_dir}")

    batch_paths: List[str] = []
    batch_secs: List[float] = []

    for path in keyframes:
        secs = extract_seconds(path, cfg.filename_seconds_regex)
        if secs is None:
            if cfg.skip_on_error:
                continue
            else:
                raise ValueError(f"Could not parse seconds from {path}")
        batch_paths.append(path)
        batch_secs.append(secs)

        if len(batch_paths) >= cfg.embed_batch_size:
            _flush_batch(batch_paths, batch_secs, start_dt, store, cfg, run_id)
            batch_paths, batch_secs = [], []

    if batch_paths:
        _flush_batch(batch_paths, batch_secs, start_dt, store, cfg, run_id)

    # Persist if supported
    if hasattr(store, 'persist'):
        store.persist(cfg.vector_store_dir)
        if cfg.log_level == 'INFO':
            print(f"Persisted vector store to {cfg.vector_store_dir}")

    return store


def _flush_batch(paths: List[str], seconds: List[float], start_dt, store, cfg: IndexingConfig, run_id: str):
    # Embed images for this batch
    img_backend, _, _ = load_embedding_backends(
        cfg.image_embedding_model, cfg.text_embedding_model, cfg.embedding_dim
    )
    embs = img_backend.embed_images(paths)

    metadatas: List[Dict[str, Any]] = []
    for p, s in zip(paths, seconds):
        real_time = None
        if start_dt is not None:
            real_time = start_dt + timedelta(seconds=s)
        md: Dict[str, Any] = {
            'video_timestamp': s,
            'abs_path': os.path.abspath(p),
            'run_id': run_id,
        }
        if real_time:
            md['real_world_time'] = real_time.isoformat()
        if cfg.extra_metadata.get('include_filename'):
            md['filename'] = os.path.basename(p)
        if cfg.extra_metadata.get('include_filesize'):
            try:
                md['filesize'] = os.path.getsize(p)
            except OSError:
                pass
        metadatas.append(md)

    store.add([os.path.basename(p) for p in paths], embs, metadatas)
    if cfg.log_level == 'DEBUG':
        print(f"Indexed batch of {len(paths)} keyframes")

__all__ = ['index_keyframes']
