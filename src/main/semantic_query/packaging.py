"""Final packaging of events + query into LLM-ready prompt."""
from __future__ import annotations
from typing import List
from .temporal import Event

PROMPT_TEMPLATE = """${SYSTEM}

USER'S QUESTION:
${QUESTION}

EVIDENCE:
${EVIDENCE}

${INSTRUCTIONS}
"""


def build_evidence(events: List[Event]) -> str:
    if not events:
        return "No events matched the query scope."
    lines = [f"Identified {len(events)} distinct events:"]
    for ev in events:
        lines.append(f"- Event {ev.event_id}: {ev.start_time} to {ev.end_time} (frames={ev.count})")
    return '\n'.join(lines)


def package_prompt(system_preamble: str, question: str, instructions: str, events: List[Event]) -> str:
    evidence = build_evidence(events)
    out = PROMPT_TEMPLATE
    out = out.replace('${SYSTEM}', system_preamble.strip())
    out = out.replace('${QUESTION}', question.strip())
    out = out.replace('${EVIDENCE}', evidence)
    out = out.replace('${INSTRUCTIONS}', instructions.strip())
    return out.strip()

__all__ = ['package_prompt']
