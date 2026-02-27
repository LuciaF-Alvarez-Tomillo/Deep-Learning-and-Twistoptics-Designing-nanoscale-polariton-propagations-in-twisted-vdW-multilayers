# TwistOptics Research Repository

TwistOptics is a research codebase to generate IFC datasets, train neural-network models, and run inverse-design inference for twisted optical multilayers.

## Requirements

- Python **3.10** (project is currently tested around this version).
- Linux/Windows: `requirements.txt`
- macOS (Apple Silicon): `requirements-mac.txt`

### Installation

```bash
# Linux / Windows
pip install -r requirements.txt

# macOS (Apple Silicon)
conda install -c apple tensorflow-deps -y
python -m pip install --upgrade pip
pip install -r requirements-mac.txt
```

## Repository layout

```text
configs/                  # YAML experiment configuration
scripts/                  # CLI entry points (dataset, train, predict)
src/twistoptics/          # Core library code
tests/                    # Basic tests
data/                     # Example/reference material data
```

## Configuration model (`configs/config.yaml`)

The configuration is intentionally split by sections:

- `paths`: shared filesystem locations used by multiple workflows.
- `system`: physical system definition (bilayer/trilayer, layer materials, substrate, superstrate permittivity).
- `data_generation`: parameters specific to dataset synthesis.
- `training`: model-training hyperparameters and dataset/model locations.
- `inference`: prediction-time options and model selection.

### Important design rule: no duplicated system definition

`system` is the **single source of truth** for:

- `bilayer`
- `materials`
- `substrate`
- `eps_superstrate`

The dataset and inference scripts automatically inherit these values.

## Typical workflow

### 1) Generate dataset

```bash
python scripts/generate_dataset.py --config configs/config.yaml
```

This reads system-level material information from `system` and generation controls from `data_generation`.

### 2) Train models

```bash
python scripts/train.py --config configs/config.yaml
```

Training uses `training` options and can reuse paths defined in `paths`.

### 3) Run inference

```bash
python scripts/predict.py --config configs/config.yaml
```

Inference uses `inference` options.


## Current default configuration (reproducing Supplementary Fig. S5)

The repository is currently configured to reproduce **Figure S5** from the supplementary material. With the checked-in `configs/config.yaml` values:

- **Dataset generation** is configured for **25,000 bilayer samples** (`n_data: 25000`, `system.bilayer: true`).
- **Training** is configured with a **fixed frequency** by using `positions: [1, 1, 1, 0]` for bilayers, where the parameter order is:
  - Bilayers: `[θ1, t1, t2, w]`
  - Trilayers: `[θ1, θ2, t1, t2, t3, w]`
  - Use `1` to indicate a parameter to predict and `0` to keep it fixed.
- **Inference** is currently configured for a **canalization study** with `N_IFCs: 10`, generating predictions for 10 IFCs that sweep all in-plane directions in **18° increments**.

These defaults are intended as a reproducible baseline; you can adapt the same workflow for other materials/systems by editing only the YAML file.

## Custom IFC studies (beyond canalization)

The current default inference workflow is configured for a **canalization study**. If you want to run a different type of IFC analysis, minimal code modifications are required.

### Custom IFC shapes

To generate IFCs different from the built-in canalization sweep:

1. Modify (or implement) the IFC-construction function inside:
```bash
src/twistoptics/utils.py
```

Add a function that generates the desired IFC geometry.

2. Update the corresponding function call in:

```bash
scripts/predict.py
```

so that inference uses your new IFC generator instead of the canalization routine.

The inference pipeline is modular: once the IFC is defined, the rest of the prediction workflow remains unchanged.

---

### Using random IFCs

If you want to test the model on **random IFCs**, this functionality is already implemented in:
```bash
scripts/predict.py
```

Currently:
- The random IFC generation lines are **commented out**.
- The canalization block is active.

To switch to random IFC testing:

1. Uncomment the random IFC generation lines.
2. Comment out the canalization-specific section.
3. Run inference as usual:

   ```bash
   python scripts/predict.py --config configs/config.yaml

## Reproducibility notes

- Dataset generation uses controlled seeds via `SEED_LIST.csv`.
- Experiment parameters are centralized in one YAML file.
- Relative paths are resolved relative to the config file location.

## Testing

Run the test suite with:

```bash
pytest
```

## License

This project is distributed under the terms in [LICENSE](LICENSE).
