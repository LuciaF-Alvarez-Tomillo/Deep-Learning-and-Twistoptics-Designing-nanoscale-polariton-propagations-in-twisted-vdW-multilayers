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
from twistoptics.data_generation import generate_data, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate IFC dataset from YAML config."
    )
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    paths = get_section(config, "paths")
    system = get_section(config, "system")
    generation = get_section(config, "data_generation")

    generation = apply_defaults(
        generation,
        {
            "bilayers": system.get("bilayer"),
            "eps_superstrate": system.get("eps_superstrate"),
            "mat": system.get("materials"),
            "mat_substrate": system.get("substrate"),
        },
    )

    output_dir = generation.pop("output_dir", None) or paths.get(
        "database", "./data/database"
    )
    output_dir = resolve_config_path(args.config, output_dir)

    # Pasar el path del archivo de semillas si está en el config
    seed_file = paths.get("seed_file")
    seed_file = resolve_config_path(args.config, seed_file)
    qs_real_m, angles, thickness, w = generate_data(
        **kwargs_for(generate_data, generation), seed_file=seed_file
    )
    save_dataset(qs_real_m, angles, thickness, w, output_dir)


if __name__ == "__main__":
    main()
