from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twistoptics.config import get_section, kwargs_for, load_config, resolve_config_path
from twistoptics.training import train_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TwistOptics models from YAML config.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = get_section(config, "paths")
    training = get_section(config, "training")

    training.setdefault("directory", paths.get("trained_models"))
    training.setdefault("database", paths.get("database"))

    for key in ("directory", "database"):
        if training.get(key) is not None:
            training[key] = resolve_config_path(args.config, training[key])

    train_models(**kwargs_for(train_models, training))

if __name__ == "__main__":
    main()
