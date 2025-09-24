Semantic Index Module
======================

Purpose:
Build an offline vector index of keyframe images, enriching each with temporal metadata.

Working Principles:
1. Keyframe Discovery: Globs filesystem paths per patterns.
2. Timestamp Extraction: Extracts video-relative seconds from filenames.
3. Embedding: Uses pluggable embedding backends (CLIP via open-clip or deterministic dummy) to produce vectors.
4. Vector Store: Persists in chosen backend (in-memory, FAISS, Chroma) with graceful fallback.
5. Metadata: Stores video_timestamp plus optional real_world_time and file attributes.

Extensibility:
- Add new embedding backends in `embedding.py`.
- Implement additional vector stores in `vector_store.py`.
- Extend metadata via `extra_metadata` in YAML config.

Failure Handling:
- Skips corrupt frames when `skip_on_error` is true.
- Falls back to in-memory store if FAISS/Chroma not installed.

Config File: `config/indexing.yml`