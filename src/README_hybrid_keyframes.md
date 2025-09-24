# Video Intelligence Pipeline (End‑to‑End Overview)

## 1. Purpose

A fully modular, config‑driven system to:
- Extract salient keyframes from raw video efficiently (hybrid motion + texture analysis).
- Build a semantic searchable index (CLIP embeddings + vector store).
- Support open‑ended natural language queries (LLM interpretation + retrieval + VLM reasoning).
- Aggregate temporal evidence into events and produce structured, LLM‑ready context.
- Orchestrate all stages through a unified pipeline and a single LLM/VLM configuration.

No domain heuristics (e.g. “morning”, “weather”, “crowd”) are hardcoded. All semantic interpretation is delegated to an LLM. Temporal reasoning relies only on metadata + interpreter outputs.

---

## 2. High‑Level Architecture

```
┌────────────────┐
│   Video File    │
└───────┬────────┘
        │ frames
        ▼
┌───────────────────────────┐
│  Hybrid Keyframe Extractor │  (Two-stage: motion → texture refinement)
└───────┬───────────────────┘
        │ keyframe images + relative timestamps
        ▼
┌───────────────────────────┐
│     Semantic Indexer       │  (Embeddings + vector DB)
└───────┬───────────────────┘
        │ vector store handle
        ▼
┌───────────────────────────┐
│     Query Orchestrator     │
│  (Interpreter + Retrieval  │
│   + VLM Reasoning + Events)│
└───────┬───────────────────┘
        │ structured prompt + events
        ▼
┌───────────────────────────┐
│ Downstream LLM Summarizer │
└───────────────────────────┘
```

---

## 3. Repository Module Map

| Layer | Path | Key Files |
|-------|------|-----------|
| Keyframe Extraction | `src/main/hybrid/` & `src/main/keyframe_extractor/` | `stages/`, YAML configs |
| Semantic Indexing | `src/main/semantic_index/` | `embedding.py`, `indexer.py`, `vector_store.py` |
| Semantic Query | `src/main/semantic_query/` | `interpreter.py`, `planner.py`, `retrieval.py`, `reasoning.py`, `temporal.py`, `packaging.py` |
| Unified Orchestration | `src/main/unified_pipeline/` | `pipeline.py`, `config/unified.yml` |
| Shared Utils | `src/main/utils/` | `llm.py`, `vlm.py`, `lbp.py`, `config_loader.py` |
| Configurations | Multiple | Stage‑scoped YAML files |

---

## 4. Stage Details

### 4.1 Hybrid Keyframe Extraction
Two‑Stage algorithm (configurable):
1. Stage 1 (Fast Motion Gate):
   - Background subtraction (MOG2) over grayscale.
   - Counts foreground pixels.
   - High motion → accept immediately.
   - Low → discard.
   - “Maybe zone” → escalate to Stage 2.
2. Stage 2 (Texture Confirmation):
   - LBP texture map + separate MOG2.
   - Validates subtle/camouflaged motion (e.g., object similar in color to background).
   - Optional morphology cleanup.
Outputs:
- Saved keyframe images (filenames encode seconds).
- List of (relative_second, filepath).
Tuning:
- Thresholds: `low_motion_threshold`, `high_motion_threshold`.
- Frame sampling (`frame_step`), resizing (`resize_scale`).
- Spacing / de‑duplication.
Configs:
- `hybrid/config/base.yml`
- `hybrid/config/stage1.yml`
- `hybrid/config/stage2.yml`
- `hybrid/config/debug.yml`

### 4.2 Semantic Indexing
Purpose: Convert keyframes into an embedding corpus for semantic search.
Steps:
1. Parse timestamp from filename (`*_XXXXs.*`).
2. Compute CLIP (open‑clip) embedding (fallback to random embedding if model missing for dry runs).
3. Add metadata:
   - `video_timestamp`
   - `real_world_time` = `start_datetime + offset`
   - `id` / `filename`
4. Store into vector backend:
   - In‑memory list (default)
   - FAISS (if installed)
   - ChromaDB (if installed)
Config (`semantic_index/config/indexing.yml`):
- `keyframe_dir`
- `start_time` (ISO)
- `model_name`
- `backend`
- `normalize_embeddings`
- `batch_size`

### 4.3 Semantic Query Pipeline
Goal: Translate a freeform user query into filtered, reasoned evidence.
Sub‑Stages:
1. Interpreter (`interpreter.py` via `utils/llm.py`):
   - LLM returns structured JSON: constraints, optional time windows, entity/action hints.
   - No built‑in heuristics; phrases like “peak hot part of the day” are resolved (or not) by LLM.
2. Planner (`planner.py`):
   - Builds QueryPlan (primary text search string + constraint bundle).
3. Retrieval (`retrieval.py`):
   - Embed query text → similarity search (top‑k).
4. Reasoning (`reasoning.py` + `utils/vlm.py`):
   - For each candidate keyframe, binary verification (e.g., “Is there a bus stand?”) using VLM (Gemini placeholder).
5. Temporal Grouping (`temporal.py`):
   - Cluster keyframe timestamps into events (gap merging).
6. Packaging (`packaging.py`):
   - Produce final evidence summary + structured prompt for summarizing LLM.

### 4.4 Unified Pipeline
`unified_pipeline/pipeline.py`:
- Loads `unified.yml`.
- Centralizes LLM config (provider, API key env var, interpreter model, reasoning model).
- Conditionally runs:
  - Keyframe extraction (if enabled)
  - Indexing (if enabled or if store absent)
  - Query execution
- Returns structured `UnifiedResult` (keyframes, store handle, query output).

---

## 5. Configuration Strategy

| File | Scope | Notes |
|------|-------|-------|
| `hybrid/config/*.yml` | Extraction micro‑tuning | Modular separation (base vs stage vs debug). |
| `semantic_index/config/indexing.yml` | Index build | Source dir, embedding backend, time anchor. |
| `semantic_query/config/query.yml` | Query behavior | Retrieval sizes, clustering gap, reasoning limits. |
| `unified_pipeline/config/unified.yml` | Orchestration | Run flags + aggregated LLM + overrides. |

All YAML remain declarative: no embedding of domain semantics (weather/time-of-day mapping, etc.).

---

## 6. LLM / VLM Integration

- Central clients live in `src/main/utils/llm.py` and `src/main/utils/vlm.py`.
- Shared config in unified YAML: `llm.provider`, `llm.api_key_env`, `llm.interpreter_model`, `llm.reasoning_model`.
- Interpreter vs reasoning differ only in prompt template + input form.
- Placeholder Gemini setup; real invocation requires:
  1. Set environment variable (e.g. `GEMINI_API_KEY`).
  2. Implement actual API call (currently stubbed & isolated for easy swap).

---

## 7. Data & Metadata Flow

1. Keyframe filenames: `frame_0123s.jpg` → numeric substring → 123.0 seconds.
2. Indexer attaches `real_world_time` for temporal semantics.
3. Query pipeline uses `video_timestamp` and `real_world_time` to cluster & describe events.
4. Final packaging enumerates events with start/end + counts.

---

## 8. Extensibility Patterns

| Need | Extension Point |
|------|------------------|
| Add new motion heuristic | Add Stage class under `hybrid/stages/` and reference in pipeline initialization. |
| Swap embedding model | Extend `EmbeddingBackend` in `semantic_index/embedding.py`. |
| Add vector backend | Implement subclass in `vector_store.py`. |
| Multi-label reasoning | Replace yes/no VLM classifier logic in `reasoning.py` / `vlm.py`. |
| Rich event descriptors | Enhance `temporal.py` to aggregate attributes (entities per cluster). |
| Additional constraints | Adjust interpreter prompt; no code change required. |

---

## 9. Typical End‑to‑End Run

1. Configure `unified_pipeline/config/unified.yml`.
2. (Optional) Populate keyframe directory if skipping extraction.
3. Run:
   ```
   python -m src.main.unified_pipeline.pipeline
   ```
4. Output:
   - Keyframes (if generated)
   - Vector store built or reused
   - Structured query result:
     - Filtered keyframes
     - Events (clustered)
     - Final prompt (ready for summarizing LLM call)

### 9.1 Minimal Keyframe Extraction via API (File‑Only Mode)

The service now exposes only a single keyframe endpoint (streaming removed):

POST /keyframes/extract
Body:
```
{ "video_path": "C:/absolute/path/to/video.mp4" }
```
Response:
```
{
   "mode": "file",
   "keyframes": [1.2, 5.6, 8.9],
   "count": 3,
   "capture_path": "C:/absolute/path/to/video.mp4"
}
```
Preconditions:
- Path must be accessible on server host.
- Any prior streaming / segment capture settings are ignored (all related code removed).

To adjust extraction behavior, edit the YAML configs under `keyframe_extractor/config/` (thresholds, frame_step, etc.).

### 9.2 Online Adaptive Thresholds

The pipeline now supports continuous (online) threshold calibration:

Concept:
- Maintain sliding window of recent Stage 1 motion scores (foreground pixel counts).
- Compute mean and std each frame; derive:
   - low_threshold = mean + k_low * std
   - high_threshold = mean + k_high * std
- Optional EMA smoothing (smooth_factor) stabilizes rapid oscillations.
- Falls back to legacy static thresholds until `min_history` scores collected.

Config (`base.yml`):
```
adaptive_threshold:
   enabled: true
   window_size: 300
   min_history: 30
   k_low: 1.0
   k_high: 3.0
   smooth_factor: 0.2
```

Disable by setting `enabled: false` (then static `low_motion_threshold` / `high_motion_threshold` in `stage1.yml` are used).

Rationale:
- Adapts to scene changes (lighting, traffic volume) without retuning.
- Automatically sets stricter thresholds in busy scenes and lower in quiet ones.

Edge Behavior:
- Ensures `high > low` by at least 1.
- If variance collapses (std≈0), thresholds compress near mean.
- Window slides; oldest scores dropped to keep responsiveness.

---

## 10. Testing & Validation (Pluggable)

- Pytest skeleton supports:
  - Vector store fallback logic
  - Planner integrity (non‑destructive query normalization)
- Future:
  - Golden event clustering tests
  - Embedding reproducibility (if deterministic seeds set)
  - LLM mock validation (JSON schema)

---

## 11. Performance Considerations

| Layer | Knobs |
|-------|-------|
| Extraction | `resize_scale`, `frame_step`, thresholds, morphology toggles |
| Indexing | Batch size, backend choice (FAISS vs in‑memory) |
| Retrieval | Top‑k balancing recall vs reasoning overhead |
| Reasoning | Limit candidates before VLM step |
| Clustering | Gap threshold affects number of events |

---

## 12. Security & Isolation

- No API keys committed: environment variable references only.
- All external interactions (LLM/VLM) isolated to two util modules.
- Fail‑open stubs allow offline development.

---

## 13. Roadmap (Suggested Enhancements)

| Feature | Description |
|---------|-------------|
| Real Gemini integration | Implement actual calls with retry + JSON schema validation. |
| Attribute extraction | Extract entities/actions for richer event summaries. |
| Temporal reasoning refinement | Overlapping window inference, duration weighting. |
| Active learning loop | Human feedback to refine reasoning filters. |
| Caching layer | Embedding & VLM result cache keyed by hash. |

---

## 14. Quick Reference Commands

Extract only (if modular script exposed):
```
python -m src.main.hybrid.pipeline
```

Index only:
```
python -c "from src.main.semantic_index.indexer import index_keyframes; import yaml; cfg=yaml.safe_load(open('src/main/semantic_index/config/indexing.yml')); index_keyframes(cfg)"
```

Run unified (full flow):
```
set UNIFIED_QUERY=vehicle movement near entrance
python -m src.main.unified_pipeline.pipeline
```

---

## 15. Minimal Integration Example (Pseudo)

```python
from src.main.unified_pipeline.pipeline import load_unified_config, run_unified
cfg = load_unified_config("src/main/unified_pipeline/config/unified.yml")
result = run_unified(cfg, user_query="crowd changes during peak hot part of the day")
print(result.query_output["prompt"])
```

---

## 16. Design Tenets Recap

- *Modular*: Each stage isolated, swappable.
- *Config‑Driven*: Behavior declaratively controlled.
- *LLM‑Centric Semantics*: No domain heuristics in code.
- *Fail‑Gracefully*: Stubs enable offline iteration.
- *Extensible*: Clear abstraction boundaries for new models / logic.

---

## 17. Support / Next Steps

If you intend to productionize:
- Add strict schema validation (e.g. Pydantic) for all YAML.
- Implement structured logging & tracing.
- Add concurrency for embedding & reasoning phases.
- Integrate observability (timings per stage).

---

*End