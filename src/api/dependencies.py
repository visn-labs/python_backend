from __future__ import annotations
from functools import lru_cache
from typing import Dict, Any
import os


def get_env_var(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


@lru_cache(maxsize=1)
def get_unified_default_config_path() -> str:
    # Could be overridden by env var
    return os.environ.get('UNIFIED_CONFIG_PATH', 'resources/unified.yml')
