"""Configuration helpers for TwistOptics scripts."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised when PyYAML is unavailable
    yaml = None


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _simple_yaml_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        key_part = line.lstrip()
        if ":" not in key_part:
            raise ValueError(f"Unsupported YAML line: {raw_line}")

        key, value = key_part.split(":", 1)
        key = key.strip()
        value = value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if value == "":
            new_dict: dict[str, Any] = {}
            current[key] = new_dict
            stack.append((indent, new_dict))
        else:
            current[key] = _parse_scalar(value)

    return root


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file into a dictionary."""
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")

    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        data = _simple_yaml_load(text)

    if not isinstance(data, dict):
        raise ValueError(f"Top-level config must be a mapping: {path}")
    return data


def get_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    """Return a config section as a dictionary, defaulting to empty."""
    value = config.get(section, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section}' must be a mapping")
    return value


def kwargs_for(func: Callable[..., Any], values: dict[str, Any]) -> dict[str, Any]:
    """Filter mapping keys to parameters accepted by ``func``."""
    accepted = set(inspect.signature(func).parameters)
    return {k: v for k, v in values.items() if k in accepted and v is not None}


def apply_defaults(values: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``values`` with missing/None entries filled from ``defaults``."""
    merged = dict(values)
    for key, default_value in defaults.items():
        if merged.get(key) is None:
            merged[key] = default_value
    return merged


def resolve_config_path(
    config_path: str | Path, maybe_relative: str | None
) -> str | None:
    """Resolve relative path values against the config file directory."""
    if maybe_relative is None:
        return None
    p = Path(maybe_relative)
    if p.is_absolute():
        return str(p)
    config_dir = Path(config_path).resolve().parent
    return str((config_dir / p).resolve())
