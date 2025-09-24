"""End-to-end semantic query pipeline (Steps 2–6).
Ties together planning, retrieval, reasoning filter, temporal clustering, and final packaging.
"""
from __future__ import annotations
import os
import yaml
from typing import Any, Dict
from .config.validation import validate_query_config
from .planner import plan_query
from .interpreter import call_llm_interpreter
from .retrieval import retrieve_candidates
from .reasoning import reasoning_filter
from .temporal import cluster_timestamps
from .packaging import package_prompt
from ..utils.llm import generate_answer
from ..semantic_index.vector_store import InMemoryVectorStore, InMemoryVectorStore as Store

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'query.yml')


def load_query_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    validate_query_config(cfg)
    return cfg


def run_semantic_query(user_query: str, store: Store, query_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = query_cfg or load_query_config()

    # Dynamic constraint extraction via interpreter (LLM or stub)
    interp_cfg = cfg['interpreter'] if 'interpreter' in cfg else {}
    constraints = {}
    if interp_cfg.get('enabled', True):  # validation ensures required keys if enabled
        constraints = call_llm_interpreter(user_query, interp_cfg)
    plan = plan_query(user_query, constraints)

    # Retrieval
    results = retrieve_candidates(
        plan,
        store,
        text_model=cfg['text_embedding_model'],
        dim=store.dim,
        top_k=cfg['initial_top_k'],
    )

    # Reasoning filter
    reasoning_cfg = cfg['reasoning'] if 'reasoning' in cfg else {}
    # Provide keyframes directory hint (reuse potential config value if present)
    keyframes_dir = cfg['keyframes_dir']
    r_provider = reasoning_cfg.get('provider', 'stub')
    if r_provider == 'stub':
        r_model = 'stub'
        r_api_env = 'stub'
    else:
        r_model = reasoning_cfg['model']
        r_api_env = reasoning_cfg['api_key_env']
    confirmed, audit = reasoning_filter(
        results,
        question=plan.reasoning_checks[0] if plan.reasoning_checks else plan.primary_text,
        enable=reasoning_cfg.get('enabled', True),
        max_items=cfg['max_confirmed'],
        provider=r_provider,
        model=r_model,
        api_key_env=r_api_env,
        keyframes_dir=keyframes_dir,
    )

    # Temporal clustering
    cluster_cfg = cfg['cluster']
    events = cluster_timestamps(
        confirmed,
        plan.time_windows,
        max_gap=cluster_cfg['max_gap_seconds'],
        min_frames=cluster_cfg['min_frames'],
    )

    # Packaging
    packaging_cfg = cfg.get('packaging', {})
    prompt = package_prompt(
        packaging_cfg.get('system_preamble', ''),
        plan.original,
        packaging_cfg.get('answer_instructions', ''),
        events,
    )

    # Final answer generation
    answer_cfg = cfg.get('answer', {})
    answer = None
    if answer_cfg.get('enabled', True):  # validation ensures required if provider != stub
        try:
            # Build confirmed frame metadata (timestamp + frame_id) for enrichment
            confirmed_frames = [
                {
                    'frame_id': c.id,
                    'timestamp': c.metadata.get('video_timestamp'),
                    'abs_path': c.metadata.get('abs_path'),
                } for c in confirmed
            ]
            answer = generate_answer(
                prompt,
                events,
                audit,
                answer_cfg,
                user_query=user_query,
                confirmed_frames=confirmed_frames,
            )
        except Exception as e:
            answer = f"(answer generation failed: {e})"

    return {
        'plan': plan,
        'retrieved': len(results),
        'confirmed': len(confirmed),
        'events': events,
        'audit': audit,
        'prompt': prompt,
        'answer': answer,
    }

__all__ = ['run_semantic_query', 'load_query_config']
