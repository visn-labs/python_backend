"""Vector store abstraction layer.
Supports multiple backends: in-memory, (placeholder) chroma, faiss.
Real implementation should wrap the chosen library.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
import os
import json

try:  # Optional FAISS
    import faiss  # type: ignore
    FaissIndexFlatIP = getattr(faiss, 'IndexFlatIP', None)
    FaissIndexFlatL2 = getattr(faiss, 'IndexFlatL2', None)
except Exception:  # pragma: no cover
    faiss = None  # type: ignore
    FaissIndexFlatIP = None
    FaissIndexFlatL2 = None

try:  # Optional Chroma
    import chromadb  # type: ignore
    from chromadb.utils import embedding_functions  # type: ignore
except Exception:  # pragma: no cover
    chromadb = None  # type: ignore
    embedding_functions = None  # type: ignore

@dataclass
class VectorRecord:
    id: str
    embedding: np.ndarray
    metadata: Dict[str, Any]

@dataclass
class InMemoryVectorStore:
    dim: int
    records: List[VectorRecord] = field(default_factory=list)

    def __len__(self):  # convenience for counting
        return len(self.records)

    def add(self, ids: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        for i, vec, meta in zip(ids, embeddings, metadatas):
            self.records.append(VectorRecord(id=i, embedding=vec.astype(np.float32), metadata=meta))

    def query(self, query_embeddings: np.ndarray, top_k: int = 10) -> List[List[VectorRecord]]:
        results: List[List[VectorRecord]] = []
        if not self.records:
            return [[] for _ in range(len(query_embeddings))]
        # Build matrix
        mat = np.vstack([r.embedding for r in self.records])  # (N, dim)
        # Normalize for cosine similarity
        mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        for q in query_embeddings:
            qn = q / (np.linalg.norm(q) + 1e-9)
            sims = mat_norm @ qn
            idx = np.argsort(-sims)[:top_k]
            results.append([self.records[int(i)] for i in idx])
        return results

    def persist(self, path: str):  # simplistic JSON + npy persistence
        os.makedirs(path, exist_ok=True)
        meta = []
        emb_list = []
        for r in self.records:
            meta.append({'id': r.id, 'metadata': r.metadata})
            emb_list.append(r.embedding)
        np.save(os.path.join(path, 'embeddings.npy'), np.vstack(emb_list))
        with open(os.path.join(path, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> 'InMemoryVectorStore':
        emb = np.load(os.path.join(path, 'embeddings.npy'))
        with open(os.path.join(path, 'meta.json'), 'r', encoding='utf-8') as f:
            meta = json.load(f)
        store = cls(dim=emb.shape[1])
        for row, vec in zip(meta, emb):
            store.records.append(VectorRecord(id=row['id'], embedding=vec, metadata=row['metadata']))
        return store


class FaissVectorStore:
    def __init__(self, dim: int, metric: str = 'ip'):
        if faiss is None or FaissIndexFlatIP is None:
            raise RuntimeError('faiss not installed')
        self.dim = dim
        if metric == 'l2' and FaissIndexFlatL2 is not None:
            self.index = FaissIndexFlatL2(dim)  # type: ignore
        else:
            self.index = FaissIndexFlatIP(dim)  # type: ignore
        self.metadata: List[Dict[str, Any]] = []
        self.ids: List[str] = []

    def add(self, ids: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        # Normalize for cosine if using IP
        if FaissIndexFlatIP is not None and isinstance(self.index, FaissIndexFlatIP):  # type: ignore[arg-type]
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
            embeddings = embeddings / norms
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadatas)
        self.ids.extend(ids)

    def query(self, query_embeddings: np.ndarray, top_k: int = 10):
        if FaissIndexFlatIP is not None and isinstance(self.index, FaissIndexFlatIP):  # cosine via normalized IP  # type: ignore[arg-type]
            norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-9
            query_embeddings = query_embeddings / norms
        D, I = self.index.search(query_embeddings.astype(np.float32), top_k)  # type: ignore[attr-defined]
        out: List[List[VectorRecord]] = []
        for row in I:
            recs: List[VectorRecord] = []
            for idx in row:
                if idx == -1:
                    continue
                recs.append(VectorRecord(id=self.ids[idx], embedding=np.zeros(self.dim, dtype=np.float32), metadata=self.metadata[idx]))
            out.append(recs)
        return out

    def persist(self, path: str):  # Minimal persistence (index + metadata)
        os.makedirs(path, exist_ok=True)
        if faiss is None:
            return
        faiss.write_index(self.index, os.path.join(path, 'faiss.index'))  # type: ignore[arg-defined]
        with open(os.path.join(path, 'faiss_meta.json'), 'w', encoding='utf-8') as f:
            json.dump({'ids': self.ids, 'metadata': self.metadata}, f)

    @classmethod
    def load(cls, path: str):
        if faiss is None:
            raise RuntimeError('faiss not installed')
        index = faiss.read_index(os.path.join(path, 'faiss.index'))
        with open(os.path.join(path, 'faiss_meta.json'), 'r', encoding='utf-8') as f:
            meta = json.load(f)
        obj = cls(dim=index.d)
        obj.index = index
        obj.ids = meta['ids']
        obj.metadata = meta['metadata']
        return obj


class ChromaVectorStore:
    def __init__(self, dim: int, storage_dir: str):
        if chromadb is None:
            raise RuntimeError('chromadb not installed')
        self.dim = dim
        self.client = chromadb.PersistentClient(path=storage_dir)  # type: ignore
        self.collection = self.client.get_or_create_collection('default')  # type: ignore

    def add(self, ids: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        self.collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas)  # type: ignore

    def query(self, query_embeddings: np.ndarray, top_k: int = 10):
        q = self.collection.query(  # type: ignore
            query_embeddings=query_embeddings.tolist(),
            n_results=top_k,
            include=["embeddings", "metadatas"],
        )
        out: List[List[VectorRecord]] = []
        ids_list = q.get('ids') or []
        meta_list = q.get('metadatas') or []
        emb_list = q.get('embeddings') or []
        for ids_row, meta_row, emb_row in zip(ids_list, meta_list, emb_list):
            recs: List[VectorRecord] = []
            for id_, md, emb in zip(ids_row, meta_row, emb_row):
                recs.append(VectorRecord(id=id_, embedding=np.array(emb, dtype=np.float32), metadata=dict(md)))
            out.append(recs)
        return out

    def persist(self, path: str):  # Persistence handled by chroma
        pass


def load_vector_store(kind: str, dim: int, storage_dir: str):
    kind = (kind or 'inmem').lower()
    if kind == 'inmem':
        return InMemoryVectorStore(dim=dim)
    if kind == 'faiss':
        if faiss is None:
            print('[vector_store] faiss not available, falling back to in-memory')
            return InMemoryVectorStore(dim=dim)
        return FaissVectorStore(dim=dim)
    if kind == 'chroma':
        if chromadb is None:
            print('[vector_store] chromadb not available, falling back to in-memory')
            return InMemoryVectorStore(dim=dim)
        return ChromaVectorStore(dim=dim, storage_dir=storage_dir)
    print(f'[vector_store] unknown kind {kind}, using in-memory')
    return InMemoryVectorStore(dim=dim)

__all__ = [
    'VectorRecord', 'InMemoryVectorStore', 'FaissVectorStore', 'ChromaVectorStore', 'load_vector_store'
]
