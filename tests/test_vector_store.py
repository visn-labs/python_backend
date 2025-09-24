import numpy as np
from src.main.semantic_index.vector_store import load_vector_store


def test_inmem_store_add_query():
    store = load_vector_store('inmem', 4, '.tmp_vs')
    vecs = np.random.rand(5,4).astype('float32')
    store.add([f'id{i}' for i in range(5)], vecs, [{"i": i} for i in range(5)])
    res = store.query(vecs[:1], top_k=3)
    assert len(res[0]) == 3


def test_unknown_kind_fallback():
    store = load_vector_store('unknown_kind', 8, '.tmp_vs')
    assert store.dim == 8
