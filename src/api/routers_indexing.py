from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import yaml
from src.main.semantic_index.indexer import index_keyframes
from src.api.routers_query import set_global_store  # reuse simple global for demonstration

router = APIRouter(prefix="/index", tags=["indexing"])

class IndexRequest(BaseModel):
    # Accept optional override for keyframes_dir if user wants to point to a fresh extraction
    keyframes_dir: str | None = None

class IndexResponse(BaseModel):
    vector_store_type: str
    dim: int
    count: int

@router.post('/build', response_model=IndexResponse, summary="Build vector index using indexing.yml configuration")
def build_index(req: IndexRequest, set_global: bool = True):
    # Directly load original indexing configuration (no unified.yml usage)
    indexing_path = os.path.join('src','main','semantic_index','config','indexing.yml')
    if not os.path.isfile(indexing_path):
        raise HTTPException(status_code=500, detail="indexing.yml configuration file missing")
    with open(indexing_path, 'r', encoding='utf-8') as f:
        raw_cfg = yaml.safe_load(f) or {}
    if req.keyframes_dir:
        raw_cfg['keyframes_dir'] = req.keyframes_dir
    keyframes_dir = raw_cfg.get('keyframes_dir')
    if not keyframes_dir or not os.path.isdir(keyframes_dir):
        raise HTTPException(status_code=404, detail=f"Keyframes dir not found: {keyframes_dir}")
    store = index_keyframes(raw_cfg)
    if set_global:
        try:
            set_global_store(store)  # type: ignore[arg-type]
        except Exception:
            pass
    # Assume store offers len via length or internal list
    # Prefer __len__ fallback to internal attributes
    count = getattr(store, 'size', None) or getattr(store, 'count', None)
    if not count:
        try:
            count = len(store)  # type: ignore
        except Exception:
            try:
                count = len(store.records)  # type: ignore[attr-defined]
            except Exception:
                count = -1
    dim = getattr(store, 'dim', -1)
    return IndexResponse(vector_store_type=store.__class__.__name__, dim=dim, count=count)
