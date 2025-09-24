"""Central config helpers for strict YAML-driven settings."""
from __future__ import annotations
from typing import Any, Dict, Iterable

class MissingConfigError(RuntimeError):
    pass

def require(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg or cfg[key] is None:
        raise MissingConfigError(f"Missing required config key: '{key}'")
    return cfg[key]

def require_nested(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur or cur[part] is None:
            raise MissingConfigError(f"Missing required config key path: '{path}' (stopped at '{part}')")
        cur = cur[part]
    return cur

def ensure_keys(section: Dict[str, Any], required: Iterable[str], section_name: str):
    for k in required:
        if k not in section or section[k] is None:
            raise MissingConfigError(f"Missing required key '{k}' in section '{section_name}'")

__all__ = [
    'MissingConfigError', 'require', 'require_nested', 'ensure_keys'
]
