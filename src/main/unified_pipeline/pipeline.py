"""Unified end-to-end pipeline orchestrating:
1. Hybrid keyframe extraction
2. Semantic indexing
3. Semantic querying

All stages share a central LLM/VLM configuration (provider + API key env).
Only prompts / model roles differ by stage (interpreter vs. reasoning).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import os
import yaml

from ..keyframe_extractor.keyframe_pipeline import HybridPipeline, load_hybrid_settings
from ..semantic_index.indexer import index_keyframes
from ..semantic_query.pipeline import run_semantic_query


@dataclass
class UnifiedResult:
    keyframes: list[float] | None
    vector_store: Any | None
    query_output: Dict[str, Any] | None


def load_unified_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _apply_llm_config(query_cfg: Dict[str, Any], llm_cfg: Dict[str, Any]):
    # Force interpreter and reasoning sections to adopt shared provider/env if present
    interp = query_cfg.setdefault('interpreter', {})
    reason = query_cfg.setdefault('reasoning', {})
    if llm_cfg.get('provider'):
        interp['provider'] = llm_cfg['provider']
        reason['provider'] = llm_cfg['provider']
    if llm_cfg.get('api_key_env'):
        interp['api_key_env'] = llm_cfg['api_key_env']
        reason['api_key_env'] = llm_cfg['api_key_env']
    if llm_cfg.get('interpreter_model'):
        interp['model'] = llm_cfg['interpreter_model']
    if llm_cfg.get('reasoning_model'):
        reason['model'] = llm_cfg['reasoning_model']


def run_unified(unified_cfg: Dict[str, Any], user_query: str | None = None) -> UnifiedResult:
    run_flags = unified_cfg.get('run', {})
    llm_cfg = unified_cfg.get('llm', {})

    keyframe_cfg = unified_cfg.get('keyframe', {})
    indexing_cfg = unified_cfg.get('indexing', {})
    query_cfg = unified_cfg.get('query', {})

    _apply_llm_config(query_cfg, llm_cfg)

    extracted = None
    store: Any | None = None
    query_output = None

    if run_flags.get('extract', True):
        # The keyframe extractor already expects a merged config directory structure.
        # Here we supply a minimal dict; HybridPipeline uses keys directly.
        kp = HybridPipeline(keyframe_cfg)
        extracted = kp.run()

    if run_flags.get('index', True):
        store = index_keyframes(indexing_cfg)

    if run_flags.get('query', True) and user_query is not None and store is not None:
        query_output = run_semantic_query(user_query, store, query_cfg)  # type: ignore[arg-type]

    return UnifiedResult(keyframes=extracted, vector_store=store, query_output=query_output)


def main():
    root = os.path.dirname(__file__)
    cfg_path = os.path.join(root, 'config', 'unified.yml')
    cfg = load_unified_config(cfg_path)
    # Example query placeholder; in real usage pass from caller.
    user_query = os.environ.get('UNIFIED_QUERY', 'describe notable events')
    res = run_unified(cfg, user_query)
    if res.query_output:
        print('Prompt:\n', res.query_output['prompt'])


if __name__ == '__main__':  # pragma: no cover
    main()
