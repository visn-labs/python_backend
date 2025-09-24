"""VLM (Vision-Language Model) yes/no verification using Google Gemini 1.5 Flash.

Replaces previous LLaVA/HF inference path. Provides a single boolean gate used by the
reasoning filter. Falls back to a permissive stub when the Gemini SDK or API key
is unavailable to avoid over-filtering results.

Environment variables:
  GEMINI_API_KEY : Google Generative AI key (required for real calls)

Provider values accepted:
  gemini (default) | stub
"""
from __future__ import annotations
from typing import Any
import os

try:  # optional dependency
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore


def _ensure_gemini(api_key_env: str = 'GEMINI_API_KEY'):  # pragma: no cover - runtime guarded
    if genai is None:
        raise RuntimeError("google-generativeai not installed; add 'google-generativeai' to requirements")
    key = os.getenv(api_key_env)
    if not key:
        raise RuntimeError(f"{api_key_env} environment variable not set")
    if hasattr(genai, 'configure'):  # type: ignore[attr-defined]
        genai.configure(api_key=key)  # type: ignore[attr-defined]


def _stub_vlm(image_path: str, question: str, **_: Any) -> bool:
    """Minimal deterministic stub decision (no domain hardcoding).

    Intentionally ignores the natural language question to avoid embedding
    semantic priors. Returns a pseudo-random but stable boolean derived from
    path digits so some frames pass and others fail, enabling downstream logic
    tests without implying actual visual understanding.
    """
    digits = ''.join(ch for ch in image_path if ch.isdigit()) or '0'
    return (int(digits[-1]) % 2) == 0


def _gemini_call(image_path: str, question: str, *, model_name: str, api_key_env: str) -> str | None:
    try:
        _ensure_gemini(api_key_env)
    except Exception as e:  # pragma: no cover
        print(f"[vlm] Gemini unavailable ({e}); using stub decision")
        return None
    if not os.path.exists(image_path):
        print(f"[vlm] image not found: {image_path}")
        return None
    try:  # Minimal multimodal invocation
        model = genai.GenerativeModel(model_name)  # type: ignore[attr-defined]
        prompt = (
            "Analyze the images thoroughly and answer strictly with Yes or No: " + question
        )
        # The SDK accepts path-like objects via opened file; supply image bytes
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        # Some SDK versions accept {'mime_type': 'image/jpeg', 'data': img_bytes}
        # Use a defensive approach.
        image_part = {"mime_type": "image/jpeg", "data": img_bytes}
        resp = model.generate_content([prompt, image_part])  # type: ignore[attr-defined]
        if hasattr(resp, 'text') and resp.text:
            return resp.text.strip()
        # Fallback: some responses embed candidates
        cand = getattr(resp, 'candidates', None)
        if cand:
            for c in cand:
                txt = getattr(getattr(c, 'content', None), 'parts', [])
                if txt:
                    maybe = str(txt[0])
                    if maybe:
                        return maybe.strip()
        return None
    except Exception as e:  # pragma: no cover
        print(f"[vlm] Gemini request failed: {e}")
        return None


def _parse_yes_no(text: str) -> bool | None:
    if not text:
        return None
    t = text.strip().lower()
    if t.startswith('yes'):
        return True
    if t.startswith('no'):
        return False
    if 'yes' in t and 'no' not in t:
        return True
    if 'no' in t and 'yes' not in t:
        return False
    return None


def vlm_yes_no(image_path: str, question: str, provider: str = 'gemini', **kwargs: Any) -> bool:
    provider = (provider or 'gemini').lower()
    if provider == 'gemini':
        if 'model' not in kwargs or 'api_key_env' not in kwargs:
            raise RuntimeError("Strict mode: 'model' and 'api_key_env' must be passed to vlm_yes_no")
        model_name = kwargs['model']
        api_key_env = kwargs['api_key_env']
        out = _gemini_call(image_path, question, model_name=model_name, api_key_env=api_key_env)
        parsed = _parse_yes_no(out or '')
        if parsed is None:
            return True  # fail-open
        return bool(parsed)
    return _stub_vlm(image_path, question)

__all__ = ["vlm_yes_no"]
