"""Vector similarity retrieval stage.
Takes a query plan, embeds primary text, queries vector store.
"""
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from ..semantic_index.embedding import load_embedding_backends
from ..semantic_index.vector_store import InMemoryVectorStore, VectorRecord


def retrieve_candidates(plan, store: InMemoryVectorStore, text_model: str, dim: int, top_k: int):
    _, txt_backend, _ = load_embedding_backends(text_model, text_model, dim)
    q_emb = txt_backend.embed_texts([plan.primary_text])[0]
    results = store.query(np.array([q_emb]), top_k=top_k)[0]
    return results

__all__ = ['retrieve_candidates']
