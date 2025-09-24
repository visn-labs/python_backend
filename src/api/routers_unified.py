from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict
import os, yaml
from src.main.keyframe_extractor.keyframe_pipeline import HybridPipeline, load_hybrid_settings
from src.main.utils.io_utils import flush_dir
from src.main.semantic_index.indexer import index_keyframes
from src.main.semantic_query.pipeline import run_semantic_query
from src.api.routers_query import set_global_store  # to allow later /query calls

router = APIRouter(prefix="/unified", tags=["unified"])

class UnifiedRequest(BaseModel):
    video_path: str = Field(..., description="Input video path to override config")
    user_query: str | None = Field(None, description="User query for semantic stage")

class UnifiedResponse(BaseModel):
    keyframes: list[float] | None
    vector_store_type: str | None
    query_output: Dict[str, Any] | None

@router.post('/run', response_model=UnifiedResponse, summary="Run full pipeline using stage YAML configurations")
def run_all(req: UnifiedRequest):
    # 1. Keyframe extraction config
    kf_config_dir = os.path.join('src','main','keyframe_extractor','config')
    if not os.path.isdir(kf_config_dir):
        raise HTTPException(status_code=500, detail="Keyframe config directory missing")
    kf_cfg = load_hybrid_settings(kf_config_dir)
    kf_cfg['video_path'] = req.video_path
    # Proactively flush output directories before pipeline run
    out_dir = kf_cfg.get('output_dir')
    if out_dir:
        flush_dir(out_dir)
    if kf_cfg.get('save_debug_masks'):
        flush_dir(kf_cfg.get('mask_output_dir', 'hybrid_keyframes/masks'))
    pipeline = HybridPipeline(kf_cfg)
    keyframes = pipeline.run()

    # 2. Indexing config
    indexing_path = os.path.join('src','main','semantic_index','config','indexing.yml')
    if not os.path.isfile(indexing_path):
        raise HTTPException(status_code=500, detail="indexing.yml configuration file missing")
    with open(indexing_path,'r',encoding='utf-8') as f:
        indexing_cfg = yaml.safe_load(f) or {}
    # Ensure indexing reads freshly generated keyframes, not stale default path
    if kf_cfg.get('output_dir'):
        indexing_cfg['keyframes_dir'] = kf_cfg['output_dir']
    store = index_keyframes(indexing_cfg)
    store_type = store.__class__.__name__
    try:
        set_global_store(store)  # type: ignore[arg-type]
    except Exception:
        pass

    query_output: Dict[str, Any] | None = None
    if req.user_query:
        query_path = os.path.join('src','main','semantic_query','config','query.yml')
        if not os.path.isfile(query_path):
            raise HTTPException(status_code=500, detail="query.yml configuration file missing")
        with open(query_path,'r',encoding='utf-8') as f:
            query_cfg = yaml.safe_load(f) or {}
        query_output = run_semantic_query(req.user_query, store, query_cfg)  # type: ignore[arg-type]

    return UnifiedResponse(keyframes=keyframes, vector_store_type=store_type, query_output=query_output)
