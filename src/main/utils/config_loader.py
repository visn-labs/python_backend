"""Config loader that merges multiple YAML files into a single dict.
Load order defines precedence (later overrides earlier)."""
from __future__ import annotations
import yaml
from typing import List, Dict, Any
import os


def load_yaml_file(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def merge_dicts(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, dict) and isinstance(result.get(k), dict):
                # recursive merge
                result[k] = merge_dicts([result[k], v])  # type: ignore[arg-type]
            else:
                result[k] = v
    return result


def load_configs(paths: List[str]) -> Dict[str, Any]:
    configs = [load_yaml_file(p) for p in paths]
    return merge_dicts(configs)

__all__ = ["load_configs"]
