from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twistoptics.config import (
    apply_defaults,
    get_section,
    kwargs_for,
    load_config,
    resolve_config_path,
)
from twistoptics.inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference from YAML config.")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths = get_section(config, "paths")
    system = get_section(config, "system")
    inference = get_section(config, "inference")

    inference = apply_defaults(
        inference,
        {
            "database": paths.get("database"),
            "mat": system.get("materials"),
            "mat_sub": system.get("substrate"),
        },
    )

    # Construir model_dir dinámicamente si solo se da la carpeta base
    model_dir = inference.get("model_dir") or paths.get("trained_models")
    model_dir = resolve_config_path(args.config, model_dir)
    num_seed = inference.get("num_seed", 3)
    nbranches = inference.get("nbranches", 1)
    # Si el path no termina en 'seed_Xbranches', lo completamos
    if not (
        f"seed_{nbranches}branches" in model_dir and f"Model_{num_seed}" in model_dir
    ):
        model_dir = f"{model_dir}/Model_{num_seed}seed_{nbranches}branches"
    inference["model_dir"] = model_dir

    for key in ("database", "model_dir"):
        if inference.get(key) is not None:
            inference[key] = resolve_config_path(args.config, inference[key])

    run_inference(**kwargs_for(run_inference, inference))


if __name__ == "__main__":
    main()
