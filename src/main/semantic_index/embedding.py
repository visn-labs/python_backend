"""Embedding abstraction for images and text.
Generic interface to allow swapping models (e.g., CLIP, SigLIP, custom transformers).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Protocol, Optional
import numpy as np

# Placeholder imports (real implementation would load actual model libs)
try:  # pragma: no cover
    import torch
    from PIL import Image
    import open_clip
except Exception:  # pragma: no cover
    torch = None
    Image = None
    open_clip = None


class ImageEmbeddingBackend(Protocol):  # Protocol for type checking
    def embed_images(self, paths: List[str]) -> np.ndarray: ...

class TextEmbeddingBackend(Protocol):
    def embed_texts(self, texts: List[str]) -> np.ndarray: ...


@dataclass
class DummyCLIPBackend(ImageEmbeddingBackend, TextEmbeddingBackend):
    dim: int = 512

    def embed_images(self, paths: List[str]) -> np.ndarray:
        # Deterministic pseudo-embedding based on filename hash (placeholder)
        vecs = []
        for p in paths:
            h = abs(hash(p)) % (10**8)
            rng = np.random.default_rng(h)
            vecs.append(rng.standard_normal(self.dim))
        return np.vstack(vecs).astype(np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            h = abs(hash(t)) % (10**8)
            rng = np.random.default_rng(h)
            vecs.append(rng.standard_normal(self.dim))
        return np.vstack(vecs).astype(np.float32)


def load_embedding_backends(image_model: str, text_model: str, dim_override: int | None = None):
    # Registry-based loader
    if open_clip is not None and torch is not None and Image is not None and image_model.startswith('openclip:'):
        model_name = image_model.split(':', 1)[1]
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
            tokenizer = open_clip.get_tokenizer(model_name)
            clip_dim = model.text_projection.shape[1] if hasattr(model, 'text_projection') else 512
        except Exception:
            clip_dim = 512
            model = None
            preprocess = None
            tokenizer = None

        if model is not None and preprocess is not None and tokenizer is not None:
            class OpenClipBackend(ImageEmbeddingBackend, TextEmbeddingBackend):  # type: ignore
                def __init__(self, model, preprocess, tokenizer, device='cpu'):
                    self.model = model.eval().to(device)
                    self.preprocess = preprocess
                    self.tokenizer = tokenizer
                    self.device = device

                def embed_images(self, paths: List[str]) -> np.ndarray:
                    imgs = []
                    for p in paths:
                        with Image.open(p) as im:  # type: ignore[union-attr]
                            imgs.append(self.preprocess(im.convert('RGB')))
                    if not imgs:
                        return np.zeros((0, clip_dim), dtype=np.float32)
                    batch = torch.stack(imgs).to(self.device)  # type: ignore[union-attr]
                    with torch.no_grad():  # type: ignore[union-attr]
                        feats = self.model.encode_image(batch)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                    return feats.cpu().numpy().astype(np.float32)

                def embed_texts(self, texts: List[str]) -> np.ndarray:
                    tokens = self.tokenizer(texts).to(self.device)
                    with torch.no_grad():  # type: ignore[union-attr]
                        feats = self.model.encode_text(tokens)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                    return feats.cpu().numpy().astype(np.float32)

            backend = OpenClipBackend(model, preprocess, tokenizer)
            return backend, backend, dim_override or clip_dim

    # Fallback dummy
    dim = dim_override or 512
    dummy = DummyCLIPBackend(dim=dim)
    return dummy, dummy, dim

__all__ = [
    'ImageEmbeddingBackend',
    'TextEmbeddingBackend',
    'DummyCLIPBackend',
    'load_embedding_backends',
]
