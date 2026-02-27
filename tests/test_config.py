from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twistoptics.config import (
    apply_defaults,
    get_section,
    kwargs_for,
    load_config,
    resolve_config_path,
)


def test_load_config_and_sections(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
paths:
  database: ./data/database
training:
  ntrain: 10
""".strip()
    )

    data = load_config(cfg)
    assert get_section(data, "paths")["database"] == "./data/database"
    assert get_section(data, "missing") == {}


def test_kwargs_for_filters_unknown_and_nulls():
    def fn(a, b=2):
        return a + b

    values = {"a": 1, "b": None, "c": 99}
    assert kwargs_for(fn, values) == {"a": 1}


def test_resolve_config_path(tmp_path):
    cfg = tmp_path / "configs" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("paths: {}")

    resolved = resolve_config_path(cfg, "./data/database")
    assert resolved.endswith("configs/data/database")


def test_apply_defaults_fills_missing_and_none_values():
    values = {"a": None, "b": 2}
    defaults = {"a": 1, "c": 3}

    assert apply_defaults(values, defaults) == {"a": 1, "b": 2, "c": 3}
