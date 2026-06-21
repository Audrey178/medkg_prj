"""Shared config loader — reads config/qa_inference.yaml once and caches."""

from pathlib import Path

import yaml

_CONFIG: dict | None = None
_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "qa_inference.yaml"


def get_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = yaml.safe_load(_CONFIG_PATH.read_text())
    return _CONFIG
