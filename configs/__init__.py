"""Configuration loader for GARIC."""

import yaml
from pathlib import Path
from typing import Any


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config, merging with defaults."""
    with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f)
        if override:
            _deep_merge(config, override)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
