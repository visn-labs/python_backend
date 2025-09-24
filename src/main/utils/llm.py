"""LLM helper utilities.

Centralizes interaction with language models for constraint interpretation.
Currently supports:
  - stub (no-op structured output)
  - gemini (placeholder call; real implementation would use google-generativeai)

No provider-specific logic is embedded elsewhere; other modules import
`interpret_constraints` only.
"""
from __future__ import annotations
from typing import Dict, Any, List
import os
import json

try:  # optional Gemini SDK
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore

STUB_RESPONSE = {
    "primary_query": None,
    "time_windows": [],
    "seasons": [],
    "weather": [],
    "environment": [],
    "entities": [],
    "actions": [],
    "other_constraints": []
}


def _ensure_gemini(api_key_env: str):  # pragma: no cover - runtime guard
    if genai is None:
        raise RuntimeError("google-generativeai not installed; add to requirements")
    key = os.getenv(api_key_env)
    if not key:
        raise RuntimeError(f"Gemini API key environment variable '{api_key_env}' not set")
    if hasattr(genai, 'configure'):  # type: ignore[attr-defined]
        genai.configure(api_key=key)  # type: ignore[attr-defined]


def _gemini_interpret(query: str, cfg: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
    api_key_env = cfg.get('api_key_env', 'GEMINI_API_KEY')
    model_name = cfg.get('model', 'gemini-1.5-pro')
    system_prompt = cfg.get('system_prompt', '')
    user_wrapper = cfg.get('user_wrapper', 'Query: "${query}"\n')
    _ensure_gemini(api_key_env)
    prompt = system_prompt + '\n' + user_wrapper.replace('${query}', query)
    # Minimal placeholder: a real call would use genai.GenerativeModel(model_name).generate_content([...])
    # For now, return stub structure with primary_query.
    resp = STUB_RESPONSE.copy()
    resp['primary_query'] = query
    return resp


def interpret_constraints(query: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    provider = cfg.get('provider', 'stub').lower()
    if provider == 'stub':
        out = STUB_RESPONSE.copy()
        out['primary_query'] = query
        return out
    if provider == 'gemini':
        return _gemini_interpret(query, cfg)
    # Unknown provider fallback
    out = STUB_RESPONSE.copy()
    out['primary_query'] = query
    return out

def generate_answer(prompt: str, events, audit, cfg: Dict[str, Any], *, user_query: str | None = None,
                    confirmed_frames: List[Dict[str, Any]] | None = None) -> str:
    """Generate final textual answer from structured evidence with generic enrichment.

    Modes:
      binary        -> yes/no style
      enumerative   -> list distinct observed concepts (delegated to VLM if provider != stub)
      descriptive   -> narrative summary
      auto          -> heuristic based on query tokens
    """
    provider = cfg.get('provider', 'stub').lower()
    api_key_env = cfg.get('api_key_env') if provider != 'stub' else None
    model = cfg.get('model') if provider != 'stub' else None
    if provider != 'stub' and (not api_key_env or not model):
        raise RuntimeError("Strict mode: answer generation requires 'model' and 'api_key_env' in config when provider != stub")
    mode_cfg = cfg.get('mode', 'auto').lower()
    max_frames_listed = int(cfg.get('max_frames_listed', 12))
    enumeration_hint = bool(cfg.get('enumeration_hint', True))
    include_events = bool(cfg.get('include_events', True))
    include_frame_table = bool(cfg.get('include_frame_table', True))

    def classify_intent(q: str | None) -> str:
        if not q:
            return 'descriptive'
        qt = q.lower()
        if mode_cfg != 'auto':
            return mode_cfg
        if any(qt.startswith(x) for x in ("is ", "are ", "was ", "were ", "do ", "does ", "did ", "has ", "have ")):
            return 'binary'
        if any(w in qt for w in ("what ", "which ", "list ", "identify", "show all")):
            return 'enumerative'
        return 'descriptive'

    intent = classify_intent(user_query)

    # Build enriched evidence text
    total_frames = sum(e.count for e in events) if events else 0
    event_lines = []
    if include_events:
        for ev in events:
            event_lines.append(
                f"Event {ev.event_id}: frame_count={ev.count} timestamps={','.join(f'{t:.2f}' for t in ev.video_timestamps[:6])}"
            )
    frame_table = []
    if include_frame_table and confirmed_frames:
        for fr in confirmed_frames[:max_frames_listed]:
            frame_table.append(f"t={fr.get('timestamp'):.2f}s id={fr.get('frame_id')}")
        if confirmed_frames and len(confirmed_frames) > max_frames_listed:
            frame_table.append(f"... (+{len(confirmed_frames)-max_frames_listed} more)")

    def stub_answer() -> str:
        # Build richer generic sections using only available structural info.
        event_span_parts: List[str] = []
        for ev in events or []:
            if ev.video_timestamps:
                start_ts = f"{ev.video_timestamps[0]:.2f}s"
                end_ts = f"{ev.video_timestamps[-1]:.2f}s"
            else:
                start_ts = end_ts = "?"
            span = start_ts if start_ts == end_ts else f"{start_ts}–{end_ts}"
            event_span_parts.append(f"E{ev.event_id}({span},{ev.count}f)")
        spans_blob = ", ".join(event_span_parts) if event_span_parts else "(none)"

        missing = [a for a in audit if 'Skipped' in a.get('decision','')]
        positives = [a for a in audit if a.get('decision') == 'Yes']
        negatives = [a for a in audit if a.get('decision') == 'No']

        def presence_phrase():
            if not events:
                return "No supporting visual evidence was detected."
            if positives and not negatives:
                return "Evidence consistently supports presence across reviewed frames."
            if positives and negatives:
                return "Mixed evidence: some frames support presence while others do not." 
            if not positives and negatives:
                return "Frames reviewed do not support presence of the queried concept."
            return "Evidence status is inconclusive."

        # Craft answer variants per intent.
        if intent == 'binary':
            if not events:
                return "No — no supporting visual evidence in the keyframes (0 events)."
            affirmative = positives and (len(positives) >= len(negatives))
            return "Yes — supported by confirmed frames." if affirmative else "No — not supported by confirmed frames."
        if intent == 'enumerative':
            if not events:
                return ("No relevant visual instances found. (Stub mode cannot extract fine-grained categories; "
                        "enable a VLM provider for object/entity enumeration.)")
            return ("Relevant instances distributed over events: "
                    f"{spans_blob}. (Stub mode: semantic categories unavailable.)")
        # descriptive / default
        if not events:
            return ("Answer: No visual evidence found.\n" 
                    "Reasoning: All candidate frames were filtered out or missing; 0 temporal events formed.")
        lines = [
            "Answer Summary: Generic structural analysis (stub mode).",
            f"Temporal Coverage: {len(events)} event(s) spanning {total_frames} keyframe(s).",
            f"Event Spans: {spans_blob}.",
            f"Presence Assessment: {presence_phrase()}",
        ]
        if missing:
            lines.append(f"Missing Frames Skipped: {len(missing)} (stale index entries).")
        lines.append("(Enable a real VLM provider for object / scene semantics and richer narrative.)")
        return "\n".join(lines)

    if provider != 'gemini':
        return stub_answer()
    try:  # pragma: no cover
        _ensure_gemini(api_key_env)  # type: ignore
        if genai is None:  # pragma: no cover
            return stub_answer()
        model_obj = genai.GenerativeModel(model)  # type: ignore[attr-defined]
        sections = [prompt]
        meta_directive = {
            'binary': "Answer strictly Yes or No followed by a short justification (<=15 words).",
            'enumerative': "List the distinct visual categories actually visible (avoid speculation).",
            'descriptive': "Provide a concise 1-2 sentence factual summary (no speculation)."
        }[intent]
        if include_events and event_lines:
            sections.append("EVENTS:\n" + "\n".join(event_lines))
        if frame_table:
            sections.append("CONFIRMED_FRAMES:\n" + "\n".join(frame_table))
        # Audit compression (only decisions for existing frames)
        audit_snips = [f"{a['frame_id']}={a['decision']}" for a in audit[:40]]
        if audit_snips:
            sections.append("AUDIT:\n" + "; ".join(audit_snips))
        if intent == 'enumerative' and enumeration_hint:
            sections.append("INSTRUCTION: If categories are unclear, describe distinguishing visual traits succinctly.")
        sections.append("RESPONSE_MODE: " + meta_directive)
        final_prompt = "\n\n".join(sections)
        resp = model_obj.generate_content(final_prompt)
        if hasattr(resp, 'text') and resp.text:
            return resp.text.strip()
        return stub_answer()
    except Exception:
        return stub_answer()

__all__ = ["interpret_constraints", "generate_answer"]