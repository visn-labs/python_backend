"""Thin wrapper preserving legacy import path for constraint interpretation.
Actual logic lives in utils.llm.interpret_constraints.
"""
from __future__ import annotations
from typing import Dict, Any
from ..utils.llm import interpret_constraints as call_llm_interpreter

__all__ = ["call_llm_interpreter"]
