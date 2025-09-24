Semantic Query Module
=====================

Purpose:
Transform a natural language user query into an evidence-grounded answer using the pre-built keyframe index.

Pipeline Stages:
1. Interpreter (LLM): Extracts structured constraints (delegated to Gemini or stub).
2. Planning: Forms a `QueryPlan` (no deterministic time parsing; temporal windows only from LLM).
3. Retrieval: Embeds primary text and performs vector similarity search.
4. Reasoning Filter (VLM): Optional frame-level yes/no verification via Gemini or stub.
5. Temporal Clustering: Groups confirmed frames into events.
6. Packaging: Produces a final prompt summarizing evidence.

Design Principles:
- No domain hardcoding (seasons, weather, time-of-day); all semantic interpretation lives in the LLM layer.
- Modular, testable components with clear interfaces.
- Graceful degradation when external services unavailable.

Configuration: `config/query.yml`.