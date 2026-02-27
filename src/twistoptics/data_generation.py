"""
===========================================================
Database generation
===========================================================
This script contains the functions to generate IFC datasets for training and evaluation.
Important! Set Rp in physics.py to the correct value for analytic IFC generation. If Rp is false,
the IFCs are generated using the analytical expression for the reflectance isofrequency contour.
The choice of method will affect the distribution and characteristics of the generated IFCs, which in turn can impact model training and performance.

Author: [Lucia F. Alvarez-Tomillo]
Date: [07/11/2025]
"""

import csv
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import twistoptics.physics as rqf


def reset_random_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(int(seed))


# def load_seed_list(seed_file: str = None):
def load_seed_list(seed_file: str | None = None):
    if seed_file is None:
        # Ruta absoluta al archivo dentro del paquete
        seed_file = str(Path(__file__).parent / "SEED_LIST.csv")
    return pd.read_csv(seed_file, header=None).values.flatten()


def generate_layer_parameters(n_layers=3, d_min=50, d_max=300):
    if n_layers == 2:
        d_layers_nm_rest = np.random.randint(d_min, d_max + 1, (2,))
        angles_deg_rest = np.random.randint(0, 181, (1,))
        return np.concatenate(([0], d_layers_nm_rest)), np.concatenate(
            ([0], angles_deg_rest)
        )
    return np.random.randint(d_min, d_max + 1, (3,)), np.random.randint(0, 181, (2,))


def plot_ifc(qs_real, al, title):
    xtm = qs_real * np.cos(al)
    ytm = qs_real * np.sin(al)
    mask = (np.abs(xtm) > 1e-10) | (np.abs(ytm) > 1e-10)
    plt.figure(figsize=(8, 8))
    plt.scatter(xtm[mask], ytm[mask], c="r", s=20)
    plt.title(title)
    plt.xlabel("qx")
    plt.ylabel("qy")
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.show()


def generate_data(
    n_data=1,
    bilayers=False,
    eps_superstrate=1.0,
    mat=None,
    mat_substrate="SiO2",
    freq_init=630,
    freq_final=680,
    d_min=50,
    d_max=300,
    n_mesh=1000,  # Number of points in the IFC mesh for optimization. The NN is structure for 1000 points, so this should not be changed without adjusting the NN architecture and training code accordingly.
    params_opt_max=None,
    plot_ifc_enabled=True,
    seed_file=None,
):
    """Generate synthetic IFC samples and associated labels.

    Returns lists compatible with the legacy CSV format: q values, angles,
    thicknesses, and frequencies.
    """

    mat = mat or ["MoO3", "MoO3", "MoO3"]
    params_opt_max = params_opt_max or [n_mesh, 2.0, 50, 50]
    t0 = time.time()
    angles, thickness, qs_real_m_list, w = [], [], [], []
    seed_list = load_seed_list(seed_file)

    for i in range(n_data):
        reset_random_seeds(seed_list[i])
        freq = int(np.random.uniform(freq_init, freq_final))
        n_layers = 2 if bilayers else 3
        d_layers_nm, angles_deg = generate_layer_parameters(n_layers, d_min, d_max)
        params = [
            d_layers_nm,
            angles_deg,
            1e4 / freq,
            eps_superstrate,
            mat_substrate,
            mat,
        ]
        ctes = rqf.Ctes_Perm(params)
        q_mode_1 = rqf.isofreq_Rp(ctes, params_opt_max)
        qs_real, _ = rqf.isofreq_minuit_Rp(ctes, q_mode_1, params_opt_max)

        if plot_ifc_enabled:
            al = np.linspace(0, 2 * np.pi, n_mesh)
            plot_ifc(qs_real, al, f"{d_layers_nm} | {angles_deg} | {freq}")

        angles.append(angles_deg.tolist())
        thickness.append(d_layers_nm.tolist())
        qs_real_m_list.append(qs_real.tolist())
        w.append([freq])

    print(f"TIME TO DATABASE OF Nº: {n_data} = {int(time.time() - t0)} s")
    return qs_real_m_list, angles, thickness, w


def save_dataset(qs_real_m, angles, thickness, w, output_dir):
    """Persist generated samples to the standard CSV dataset layout."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        output / "angles.csv": angles,
        output / "thickness.csv": thickness,
        output / "qs_real_m.csv": qs_real_m,
        output / "freq.csv": w,
    }
    for file_name, data in files.items():
        mode = "a" if file_name.exists() else "w"
        with open(file_name, mode=mode, newline="") as archivo:
            csv.writer(archivo).writerows(data)


# def main():
#     """CLI-style entry point used by ``scripts/generate_dataset.py``."""
#     mat = ["MoO3", "MoO3", "MoO3"]
#     qs_real_m, angles, thickness, w = generate_data(
#         n_data=10, mat=mat, plot_ifc_enabled=True
#     )
#     save_dataset(qs_real_m, angles, thickness, w, f"Database_trilayers_{'_'.join(mat)}")


# if __name__ == "__main__":
#     main()
