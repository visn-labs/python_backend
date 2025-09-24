from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict
import os, yaml
from src.main.semantic_query.pipeline import run_semantic_query
from src.main.semantic_index.vector_store import InMemoryVectorStore

router = APIRouter(prefix="/query", tags=["semantic-query"])

class QueryRequest(BaseModel):
    user_query: str = Field(..., description="Natural language question")

class QueryResponse(BaseModel):
    retrieved: int
    confirmed: int
    events: Any
    prompt: str

@router.post('/', response_model=QueryResponse, summary="Run semantic query using YAML config")
def semantic_query(req: QueryRequest):
    global GLOBAL_VECTOR_STORE
    if GLOBAL_VECTOR_STORE is None:
        raise HTTPException(status_code=400, detail="Vector store not initialized. Run unified or indexing first.")
    query_path = os.path.join('src','main','semantic_query','config','query.yml')
    if not os.path.isfile(query_path):
        raise HTTPException(status_code=500, detail="query.yml configuration file missing")
    with open(query_path,'r',encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    res = run_semantic_query(req.user_query, GLOBAL_VECTOR_STORE, cfg)
    return QueryResponse(retrieved=res['retrieved'], confirmed=res['confirmed'], events=res['events'], prompt=res['prompt'])

# Simple mutable global for demo purposes (not prod-safe)
GLOBAL_VECTOR_STORE: InMemoryVectorStore | None = None

def set_global_store(store: InMemoryVectorStore):  # utility for other routers
    global GLOBAL_VECTOR_STORE
    GLOBAL_VECTOR_STORE = store
    return True
