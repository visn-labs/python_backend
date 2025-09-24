"""Natural language query planner.
Converts free-form user query into a structured retrieval & reasoning plan.
Deterministic time parsing removed: ALL temporal interpretation (even explicit '09:00-10:00')
must come from the LLM interpreter constraints; planner itself is pass-through.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class QueryPlan:
    original: str
    primary_text: str
    reasoning_checks: List[str]
    time_windows: List[tuple[str, str]]  # list of (start, end) in HH:MM:SS
    constraints: Dict[str, Any]          # structured constraints (entities, actions, seasons, etc.)

DEFAULT_REASONING_QUESTIONS = ["Is the queried concept clearly present?"]


def plan_query(user_query: str, interpreter_constraints: Dict[str, Any] | None = None) -> QueryPlan:
    q = user_query.strip()
    primary = q  # no stripping of time expressions

    # Interpreter-provided windows (list of dicts with start/end)
    interp_windows: List[tuple[str, str]] = []
    if interpreter_constraints and 'time_windows' in interpreter_constraints:
        for tw in interpreter_constraints.get('time_windows', []) or []:
            start = tw.get('start')
            end = tw.get('end')
            if start and end:
                interp_windows.append((start, end))

    windows = interp_windows  # ONLY interpreter-provided windows

    plan = QueryPlan(
        original=q,
        primary_text=primary,
        reasoning_checks=DEFAULT_REASONING_QUESTIONS.copy(),
        time_windows=windows,
        constraints=interpreter_constraints or {},
    )
    return plan

__all__ = ['QueryPlan', 'plan_query']
