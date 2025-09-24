"""Validation for semantic query configuration (strict mode)."""
from __future__ import annotations
from typing import Dict, Any
from ...utils.config import MissingConfigError, ensure_keys

def validate_query_config(cfg: Dict[str, Any]) -> None:
    # Root level required keys
    root_required = [
        'initial_top_k', 'max_confirmed', 'keyframes_dir'
    ]
    for k in root_required:
        if k not in cfg or cfg[k] is None:
            raise MissingConfigError(f"Missing required root key '{k}'")
    # Cluster
    cluster = cfg.get('cluster')
    if not isinstance(cluster, dict):
        raise MissingConfigError("Missing 'cluster' section")
    ensure_keys(cluster, ['max_gap_seconds', 'min_frames'], 'cluster')

    # Interpreter section (conditional)
    interp = cfg.get('interpreter') or {}
    if interp.get('enabled', True):
        if 'provider' not in interp:
            raise MissingConfigError("interpreter.provider required when interpreter.enabled is true")
        if interp['provider'] != 'stub':
            ensure_keys(interp, ['model', 'api_key_env'], 'interpreter')

    # Reasoning section (conditional)
    reasoning = cfg.get('reasoning') or {}
    if reasoning.get('enabled', True):
        if 'provider' not in reasoning:
            raise MissingConfigError("reasoning.provider required when reasoning.enabled is true")
        if reasoning['provider'] != 'stub':
            ensure_keys(reasoning, ['model', 'api_key_env'], 'reasoning')

    # Answer section (conditional)
    answer = cfg.get('answer') or {}
    if answer.get('enabled', True):
        if 'provider' not in answer:
            raise MissingConfigError("answer.provider required when answer.enabled is true")
        if answer['provider'] != 'stub':
            ensure_keys(answer, ['model', 'api_key_env'], 'answer')

__all__ = ['validate_query_config']
