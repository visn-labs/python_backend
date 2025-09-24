Utils Module
============

Purpose:
Central collection of helper utilities shared across modules (config loading, LLM + VLM adapters, image morphology, feature encoding).

Contents:
- `config_loader.py`: YAML loading & validation helpers.
- `io_utils.py`: File I/O primitives.
- `lbp.py`: Local Binary Pattern feature computation.
- `morph.py`: Morphological image operations.
- `llm.py`: Constraint interpretation (Gemini / stub).
- `vlm.py`: Vision-language yes/no verification (Gemini / stub).

Guidelines:
- Keep pure functions & thin wrappers here; domain pipelines should import from utils to avoid duplication.
- Provider-specific logic should remain encapsulated behind a narrow function surface (e.g., `interpret_constraints`, `vlm_yes_no`).