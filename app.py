from __future__ import annotations
from fastapi import FastAPI
from src.api.routers_keyframes import router as keyframes_router
from src.api.routers_indexing import router as indexing_router
from src.api.routers_query import router as query_router, set_global_store
from src.api.routers_unified import router as unified_router

app = FastAPI(title="Insights Video Pipeline API", version="0.1.0")

app.include_router(keyframes_router)
app.include_router(indexing_router)
app.include_router(query_router)
app.include_router(unified_router)

@app.get('/health')
async def health():
    return {"status": "ok"}
