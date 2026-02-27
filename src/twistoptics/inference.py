"""
===========================================================
Prediction
===========================================================
This script contains the functions to run inference with a trained model,
including loading the model, preparing synthetic IFC inputs, running predictions,
selecting best candidates using an oracle distance metric, and saving comparison plots.

Author: [Lucia F. Alvarez-Tomillo]
Date: [07/11/2025]
"""

import os
import time
from functools import partial
from pathlib import Path
import numpy as np
from tensorflow.keras import models

import twistoptics.utils as auxf


def run_inference(
    model_dir="./Models_Trained_bilayers_w_fixed",
    database="./Database_bilayers_MoO3_MoO3_MoO3",
    num_seed=3,
    nbranches=1,
    positions=None,
    input2=None,
    N_IFCs=10,
    N_best_predictions=3,
    mat=["MoO3", "MoO3", "MoO3"],
    mat_sub="SiO2",
):
    """Load a trained model and produce IFC parameter predictions.

    The function reproduces the original prediction pipeline: build synthetic IFC
    inputs, run multi-branch inference, select best candidates using the oracle
    distance metric, and save comparison plots to ``resultsCanalization``.
    """
    t1 = time.time()
    positions = positions or [1, 1, 1, 0]
    nfeatures = sum(positions)
    if len(positions) != 4:
        bilayer = False
        ref_all = np.array([1, 1, 1, 1, 1, 1])
    else:
        bilayer = True
        ref_all = np.array([1, 1, 1, 1])

    positions_input = ref_all - np.array(positions)
    shape_input_2 = int(np.sum(positions_input))

    model_path = Path(model_dir) / f"Model_{num_seed}seed_{nbranches}branches.h5"
    results_dir = Path(model_dir) / "results"
    os.makedirs(results_dir, exist_ok=True)

    model = models.load_model(
        model_path,
        custom_objects={"NBsym_loss": auxf.symmetry_loss_Nb(nbranches, nfeatures)},
        compile=False,
    )

    # Canalization Study
    # =============================================================================
    # qs_real is the array of q values for the real IFC, the input for the prediction.
    # =============================================================================
    qs_real, thetas_real_m = auxf.canalization(4.5)
    qs_real = np.where(qs_real > 30, 0, qs_real)
    thetas_real_m = np.linspace(0, 2 * np.pi, 1000)

    for i in range(N_IFCs):
        input2 = input2 or None
        qs_real_m = auxf.rotate_array(
            qs_real, 50 * i
        )  # Comment if you want to use the same IFC for all
        #  predictions or generate a new one each time. The rotation is just to have different IFCs for the same parameters,
        #  as the rotation does not change the physics of the system.

        # SYSTEM PARAMETERS (random IFCs)

        # freq = int(np.random.uniform(630, 680))#1e7/1200, 1e7/600))#910,930)) (280,380))
        # wavelength_mu = 1e4/freq #microns
        # d_layers_nm = np.random.randint(50, 300+1, (3,)); #d_layers_nm = np.concatenate(([0], d_layers_nm))
        # angles_deg = np.random.randint(0, 180+1, (2,)); #angles_deg = np.concatenate(([0], angles_deg))

        # wavelength_mu = 1e4 / freq  # microns
        # params = [d_layers_nm, angles_deg, wavelength_mu, eps_superstrate, subs, mat]
        # paramslist = [d_layers_nm, angles_deg, freq]
        # print("The data for the prediction is: ", paramslist)
        # ctes = rqf.Ctes_Perm(params)
        # q_mode_1 = isofreq_sem(ctes)
        # qs_real_m, qs_imag_m = isofreq(ctes, q_mode_1)

        qx_real = qs_real_m * np.cos(thetas_real_m)
        qy_real = qs_real_m * np.sin(thetas_real_m)

        if shape_input_2 == 0:
            angles_thickness, _ = auxf.predict_No_input2(
                model, qs_real_m, nbranches, database, bilayer
            )
        else:
            angles_thickness, _ = auxf.predict_input2(
                model,
                qs_real_m,
                input2,
                shape_input_2,
                positions,
                nbranches,
                database,
                bilayer,
            )

        (
            array_best_parameters,
            array_parameters,
            list_distances,
            q_nn_min,
            q_nn_list_mins,
            params_min,
            params_list_mins,
        ) = auxf.y_NN_min_oracle(
            angles_thickness,
            positions,
            positions_input,
            qs_real_m,
            input2,
            nbranches,
            nfeatures,
            N_best_predictions,
            mat,
            mat_sub,
            bilayer,
        )

        qx_real_nn = q_nn_min * np.cos(thetas_real_m)
        qy_real_nn = q_nn_min * np.sin(thetas_real_m)
        auxf.plot_best(
            [qx_real, qx_real_nn],
            [qy_real, qy_real_nn],
            params_min,
            results_dir,
            title=i,
        )
        auxf.plot_multiple(
            qx_real,
            qy_real,
            q_nn_list_mins,
            thetas_real_m,
            params_list_mins,
            results_dir,
            cota=30,
            num_ifc=N_best_predictions,
            num_ifcs=i,
        )

    print(f"Total execution time: {time.time() - t1:.2f} seconds")


# def main():
#     """CLI-style entry point used by ``scripts/predict.py``."""
#     run_inference()


# if __name__ == "__main__":
#     main()
