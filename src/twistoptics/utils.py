# -*- coding: utf-8 -*-
"""
===========================================================
Aux_Functions_Rp
===========================================================
This script contains auxiliary functions used in the document.

Author: [Lucia F. Alvarez-Tomillo]
Date: [07/11/2025]
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional

# print(tf.__version__)
import os
import sys
import twistoptics.physics as rqf
from functools import partial
import warnings
import csv


def guardar_datos_en_csv(datos, nombre_archivo):
    """
    Save a list of rows to a CSV file.

    Parameters
    ----------
    datos : list of list
        Data to be saved, each sublist is a row.
    nombre_archivo : str
        Path to the output CSV file.
    """
    with open(nombre_archivo, mode="w", newline="") as archivo:
        escritor_csv = csv.writer(archivo, quoting=csv.QUOTE_ALL)
        for fila in datos:
            escritor_csv.writerow(fila)


# Custom warning class
class MiAdvertencia(Warning):
    pass


# Custom warning renderer
def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    """
    Custom warning display function that prints warnings in red.
    """
    red = "\033[91m"
    reset = "\033[0m"
    sys.stderr.write(f"{red}{message}{reset}\n")


warnings.showwarning = custom_showwarning


# =============================================================================
# Normalization and data base
# =============================================================================
def read_single_csv(file: str) -> np.ndarray:
    """
    Read a CSV file without header and return as ndarray.
    """
    p = Path(file)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file}")
    X = pd.read_csv(p, header=None, dtype=float)
    return X.to_numpy()


def normalize_data(data, min_val, max_val):
    """
    Normalize data to [0, 1] range.
    """
    return (data - min_val) / (max_val - min_val)


def unnormalize_data(data_norm, min_val, max_val):
    """
    Undo normalization to original scale.
    """
    return data_norm * (max_val - min_val) + min_val


def load_database(database):  # normalize_database
    file_angles = database + "/angles.csv"
    file_thickness = database + "/thickness.csv"
    file_q_real = database + "/qs_real_m.csv"
    file_freq = database + "/freq.csv"
    # Check if the first column of angles.csv is all zeros (bilayer)
    if np.all(read_single_csv(file_angles)[:, 0] == 0):
        angles_dataset_ref = read_single_csv(file_angles)[:, 1:]
        thickness_dataset_ref = read_single_csv(file_thickness)[:, 1:]
    else:
        angles_dataset_ref = read_single_csv(file_angles)
        thickness_dataset_ref = read_single_csv(file_thickness)
    q_real_dataset_ref = read_single_csv(file_q_real)
    freq_dataset_ref = read_single_csv(file_freq)

    return (
        angles_dataset_ref,
        thickness_dataset_ref,
        q_real_dataset_ref,
        freq_dataset_ref,
    )


def prepare_data(ntrain, nvalidation, positions, database):
    """
    Prepare normalized training and validation data.

    Parameters
    ----------
    ntrain : int
        Number of training samples.
    nvalidation : int
        Number of validation samples.
    positions : list of int
        Mask indicating which features are predicted (1) or input (0).
    database : str
        Path to the dataset folder.

    Returns
    -------
    xtr : dict
        Training input dictionary.
    xva : dict
        Validation input dictionary.
    ytr_single : np.ndarray
        Training targets.
    yva_single : np.ndarray
        Validation targets.
    indices : np.ndarray
        Indices of q_real > 50.
    """
    angles, thickness, q_real, freq = load_database(database)
    ndata_t = ntrain + nvalidation

    indices = np.argwhere(q_real > 50)

    ndata = q_real.shape[0]
    q_size = q_real.shape[1]
    if ndata > ndata_t:
        warnings.warn("You are not using the entire database", MiAdvertencia)
    elif ndata < ndata_t:
        raise ValueError(
            "Error: The database is smaller than the sum of the selected training and validation values."
        )
    else:
        print(
            "The database matches the sum of examples used in validation and training."
        )

    min_q, max_q = np.amin(q_real), np.amax(q_real)
    min_angles, max_angles = np.amin(angles), np.amax(angles)
    min_thickness, max_thickness = np.amin(thickness), np.amax(thickness)
    min_freq, max_freq = np.amin(freq), np.amax(freq)
    # print(min_q, max_q, min_angles, max_angles, min_thickness, max_thickness, min_freq, max_freq)
    angles_norm = normalize_data(angles, min_angles, max_angles)[0:ndata_t, :]
    thickness_norm = normalize_data(thickness, min_thickness, max_thickness)[
        0:ndata_t, :
    ]
    freq_norm = normalize_data(freq, min_freq, max_freq)[0:ndata_t, :]
    q_real_norm = normalize_data(q_real, min_q, max_q)[0:ndata_t, :]
    q_real_norm = np.reshape(q_real_norm, (ndata_t, q_size, 1))

    complete = np.concatenate([angles_norm, thickness_norm, freq_norm], axis=1)

    if len(positions) != complete.shape[1]:
        raise ValueError(
            f"The length of 'positions' ({len(positions)}) does not match the number of features ({complete.shape[1]})."
        )

    mask_x = np.array(positions) == 0
    mask_y = np.array(positions) == 1

    parameters_x = complete[:, mask_x]
    parameters_y = complete[:, mask_y]

    if mask_x.any():
        xtr = {
            "input0": q_real_norm[0:ntrain, :500, :],
            "input2": parameters_x[0:ntrain, :],
        }
        xva = {
            "input0": q_real_norm[ntrain : ntrain + nvalidation, :500, :],
            "input2": parameters_x[ntrain : ntrain + nvalidation, :],
        }
    else:
        xtr = {"input0": q_real_norm[0:ntrain, :500, :]}
        xva = {"input0": q_real_norm[ntrain : ntrain + nvalidation, :500, :]}
        print("There is no input 2")
    ytr_single = parameters_y[0:ntrain, :]
    yva_single = parameters_y[ntrain : ntrain + nvalidation, :]

    return xtr, xva, ytr_single, yva_single, indices


# =============================================================================
# Network training functions
# =============================================================================


def load_seed_list(seed_file: str = "SEED_LIST.csv"):
    seed_path = Path(seed_file)
    seed_list = pd.read_csv(seed_path, header=None)
    seed_list = np.array(seed_list)
    max_num = np.shape(seed_list)[0]
    seed_list = np.reshape(seed_list, (max_num,))
    return seed_list


def reset_random_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(int(seed))


def symmetry_loss_Nb(Nbranches, Nfeatures):
    @tf.function
    def NBsym_loss(x_true, x_pred):
        mbs = tf.shape(x_pred)[0]

        # Cast tensors to float32 for stable TensorFlow operations
        x_true = tf.cast(x_true, dtype="float32")
        x_pred = tf.cast(x_pred, dtype="float32")

        diff_squared = (x_pred - x_true) * (x_pred - x_true)
        # Convolution kernel must span exactly one branch-worth of features
        sum_kernel = tf.constant(np.ones((1, Nfeatures)), tf.float32)
        # Use 2D convolution to aggregate feature-wise errors per branch
        cost_matrix = tf.nn.conv2d(
            tf.reshape(diff_squared, (1, mbs, int(Nfeatures * Nbranches), 1)),
            tf.reshape(sum_kernel, (1, Nfeatures, 1, 1)),
            strides=[1, Nfeatures],
            padding="VALID",
        )
        cost_matrix = tf.reshape(cost_matrix, (mbs, Nbranches))

        cost_matrix = 0.5 * cost_matrix
        cost_vector = tf.math.reduce_min(cost_matrix, axis=1)
        mbs = tf.cast(mbs, dtype="float32")
        loss = tf.math.reduce_sum(cost_vector) / mbs

        return loss

    return NBsym_loss


def expand_outputs_for_branches(y_single, Nbranches):
    y = np.tile(y_single, reps=(1, Nbranches))
    return y


def save_history(history_keras, path_hist):
    history_dict = history_keras.history
    training_cost = history_dict["loss"]  # Store training metric values for each epoch
    evaluation_cost = history_dict["val_loss"]

    epochs = len(evaluation_cost)  # Total number of epochs recorded
    xx = np.linspace(0, epochs - 1, epochs)  # Epoch index array [0, ..., epochs-1]

    f1 = open(path_hist, "w")
    for i in range(0, epochs):  # Serialize metrics row-by-row as text
        summary = (
            str(xx[i])
            + " "
            + str(evaluation_cost[i])
            + " "
            + " "
            + str(training_cost[i])
            + " "
            + ""
            + "\n"
        )
        f1.write(summary)

    f1.close()


def save_metrics_classifier(history_keras, path_hist):
    history_dict = history_keras.history
    training_cost = history_dict[
        "categorical_accuracy"
    ]  # Store training metric values for each epoch
    evaluation_cost = history_dict["val_categorical_accuracy"]

    epochs = len(evaluation_cost)  # Total number of epochs recorded
    xx = np.linspace(0, epochs - 1, epochs)  # Epoch index array [0, ..., epochs-1]

    f1 = open(path_hist, "w")
    for i in range(0, epochs):  # Serialize metrics row-by-row as text
        summary = (
            str(xx[i])
            + " "
            + str(evaluation_cost[i])
            + " "
            + " "
            + str(training_cost[i])
            + " "
            + ""
            + "\n"
        )
        f1.write(summary)

    f1.close()


# Remove branch with least minimum predictions
def pop_several(my_list, *idx_remove):
    sorted_idx_remove = sorted(idx_remove, reverse=False)
    for idx in sorted_idx_remove:
        my_list.pop(idx)


def remove_min_branch(weights, x_true, x_pred, Nbranches, Nfeatures):
    ndata = np.shape(x_true)[0]

    cost_matrix = (x_pred[:, 0] - x_true[:, 0]) * (x_pred[:, 0] - x_true[:, 0])
    for j in range(1, Nfeatures):
        cost_matrix = cost_matrix + (x_pred[:, j] - x_true[:, j]) * (
            x_pred[:, j] - x_true[:, j]
        )

    for i in range(1, Nbranches):
        cost_column = (x_pred[:, Nfeatures * i] - x_true[:, 0]) * (
            x_pred[:, Nfeatures * i] - x_true[:, 0]
        )
        for j in range(1, Nfeatures):
            cost_column = cost_column + (
                x_pred[:, Nfeatures * i + j] - x_true[:, j]
            ) * (x_pred[:, Nfeatures * i + j] - x_true[:, j])

        cost_matrix = np.concatenate((cost_matrix, cost_column), axis=0)

    cost_matrix = np.reshape(cost_matrix, (Nbranches, ndata))
    cost_matrix = np.transpose(cost_matrix)
    cost_matrix = 0.5 * cost_matrix

    ##### CALCULATE column_mins #####
    # For each data in training_data_set, it gives the index of the branch that gives the minimum loss
    column_mins = list(np.argmin(cost_matrix, axis=1))

    ##### CALCULATE list_counter #####
    # For each branch, it gives the number of data for which the branch predicts with minimum loss
    list_counter = []
    for i in range(Nbranches):
        list_counter.append(column_mins.count(i))

    # Identify the branch with the fewest minimum-loss assignments
    # and compute which weight/bias tensors must be removed.

    idx_min = list_counter.index(min(list_counter))
    print("index min = ", idx_min)

    # INDICES FROM ELEMENTS TO BE REMOVED IN NEW_WEIGHTS
    w_layer1 = int(-2 - 2 * (2 * Nbranches - 1 - idx_min))
    b_layer1 = int(-1 - 2 * (2 * Nbranches - 1 - idx_min))
    w_layer2 = int(-2 - 2 * (Nbranches - 1 - idx_min))
    b_layer2 = int(-1 - 2 * (Nbranches - 1 - idx_min))

    print("REMOVED INDICES:")
    print(w_layer1)
    print(b_layer1)
    print(w_layer2)
    print(b_layer2)

    pop_several(weights, w_layer1, b_layer1, w_layer2, b_layer2)

    return list_counter


# =============================================================================
# Post-processing predictions
# =============================================================================


def _get_normalization_arrays(database, bilayer, positions, Nbranches):
    """
    Helper to compute all normalization arrays and min/max for q and features.

    Args:
        database (str): Path to the dataset folder.
        bilayer (bool): True if bilayer ordering is used; False for trilayer.
        positions (list or None): Mask for which features are predicted (1) or input (0).
        Nbranches (int): Number of output branches.

    Returns:
        dict: Contains min_q, max_q, min_arr_input2, max_arr_input2, min_arr_pred_br, max_arr_pred_br.
    """
    angles, thickness, q_real, freq = load_database(database)
    min_q, max_q = np.amin(q_real), np.amax(q_real)
    min_angles, max_angles = np.amin(angles), np.amax(angles)
    min_thickness, max_thickness = np.amin(thickness), np.amax(thickness)
    min_freq, max_freq = np.amin(freq), np.amax(freq)

    if bilayer:
        min_complete = np.array([min_angles, min_thickness, min_thickness, min_freq])
        max_complete = np.array([max_angles, max_thickness, max_thickness, max_freq])
    else:
        min_complete = np.array(
            [
                min_angles,
                min_angles,
                min_thickness,
                min_thickness,
                min_thickness,
                min_freq,
            ]
        )
        max_complete = np.array(
            [
                max_angles,
                max_angles,
                max_thickness,
                max_thickness,
                max_thickness,
                max_freq,
            ]
        )

    if positions is not None:
        pos_arr = np.array(positions)
        mask_x = pos_arr == 0
        mask_y = pos_arr == 1
        min_arr_input2 = np.array([min_complete[mask_x]])
        max_arr_input2 = np.array([max_complete[mask_x]])
        min_arr_pred_single = np.array([min_complete[mask_y]])
        max_arr_pred_single = np.array([max_complete[mask_y]])
    else:
        min_arr_input2 = None
        max_arr_input2 = None
        min_arr_pred_single = np.array([min_complete])
        max_arr_pred_single = np.array([max_complete])

    min_arr_pred_br = np.concatenate(
        [min_arr_pred_single for _ in range(Nbranches)], axis=1
    )
    max_arr_pred_br = np.concatenate(
        [max_arr_pred_single for _ in range(Nbranches)], axis=1
    )

    return {
        "min_q": min_q,
        "max_q": max_q,
        "min_arr_input2": min_arr_input2,
        "max_arr_input2": max_arr_input2,
        "min_arr_pred_br": min_arr_pred_br,
        "max_arr_pred_br": max_arr_pred_br,
    }


def _normalize_inputs(q_real, input2, norm_arrays):
    """
    Helper to normalize q_real and input2 using normalization arrays.

    Args:
        q_real (np.ndarray): 1D array of q samples (length N).
        input2 (np.ndarray or None): Array of scalar inputs to normalize, or None.
        norm_arrays (dict): Output of _get_normalization_arrays.

    Returns:
        tuple: (q_real_norm, input2_norm)
    """
    N = q_real.shape[0]
    q_half = q_real[: N // 2].reshape(N // 2, 1)
    q_real_norm = normalize_data(q_half, norm_arrays["min_q"], norm_arrays["max_q"])
    input2_norm = None
    if input2 is not None and norm_arrays["min_arr_input2"] is not None:
        input2_norm = normalize_data(
            input2, norm_arrays["min_arr_input2"], norm_arrays["max_arr_input2"]
        )
    return q_real_norm, input2_norm


def predict_input2(
    model,
    q_real,
    input2,
    shape_input_2,
    positions,
    Nbranches,
    database,
    bilayer,
):
    """
    Predict outputs given q_real and input2.
    """
    norm_arrays = _get_normalization_arrays(database, bilayer, positions, Nbranches)
    q_real_norm, input2_norm = _normalize_inputs(q_real, input2, norm_arrays)
    angles_deg_norm_NN = model.predict(
        {
            "input0": np.reshape(q_real_norm, (1, q_real_norm.shape[0], 1)),
            "input2": np.reshape(input2_norm, (1, shape_input_2)),
        }
    )
    angles_thickness_NN = unnormalize_data(
        angles_deg_norm_NN,
        norm_arrays["min_arr_pred_br"],
        norm_arrays["max_arr_pred_br"],
    )
    return angles_thickness_NN, angles_deg_norm_NN


def predict_No_input2(model, q_real, Nbranches, database, bilayer=True):
    """
    Predict outputs given only q_real (no input2), using shared normalization logic.
    """
    norm_arrays = _get_normalization_arrays(
        database, bilayer, positions=None, Nbranches=Nbranches
    )
    q_real_norm, _ = _normalize_inputs(q_real, None, norm_arrays)
    angles_thickness_norm_NN = model.predict(
        np.reshape(q_real_norm, (1, q_real_norm.shape[0], 1))
    )
    angles_thickness_NN = unnormalize_data(
        angles_thickness_norm_NN,
        norm_arrays["min_arr_pred_br"],
        norm_arrays["max_arr_pred_br"],
    )
    return angles_thickness_NN, angles_thickness_norm_NN


# =============================================================================
# Oracle distance and loss functions
# =============================================================================


# Mean squared error
def MSE_IFC(q_true, q_NN):
    N = np.shape(q_true)[0]
    return np.sum((q_true - q_NN) * (q_true - q_NN)) / (N)


def distance_q(q_true, q_pred):
    q_min = 2.0  # minimum physically meaningful q value
    q_cut = 50.0  # artificial plotting cutoff used in generated data
    delta_cut = 5.0  # tolerance band around the plotting cutoff
    zero_budget = 0.10  # allowed fraction of zeros without penalty
    zero_total_budget = 0.40  # maximum reasonable fraction of total zeros

    alpha = 2.0  # penalty weight
    beta_sparse = 1.5  # penalty weight for overly sparse IFC predictions

    q_true = np.asarray(q_true)
    q_pred = np.asarray(q_pred)
    N = len(q_true)

    # --- Physical validity masks ---
    valid_mask = (q_true >= q_min) & (q_pred >= q_min)

    # Near-zero values in the prediction
    zero_pred = q_pred < q_min

    # Zeros that should not be penalized
    zero_allowed = (
        (q_true < q_min)  # target has no physical branch
        | (q_true >= q_cut - delta_cut)  # artificial cutoff zone
    )

    # Zeros that should be penalized
    zero_bad = zero_pred & (~zero_allowed)

    # --- 1) Shape mismatch term ---
    if np.any(valid_mask):
        dist_shape = np.mean(np.abs(q_true[valid_mask] - q_pred[valid_mask]))
    else:
        dist_shape = q_cut

    # --- 2) Penalty for unjustified zeros ---
    frac_bad_zeros = np.sum(zero_bad) / max(N, 1)
    # print("\n")
    # print(f"Non-physical zeros: {frac_bad_zeros:.2%}")
    # print("\n")
    if frac_bad_zeros > zero_budget:
        penalty_bad = np.mean((q_true[zero_bad] - q_pred[zero_bad]) ** 2)
    else:
        penalty_bad = 0.0

    # --- 3) Penalty for overly sparse IFCs ---
    frac_total_zeros = np.sum(zero_pred) / max(N, 1)

    if frac_total_zeros > zero_total_budget:
        penalty_sparse = (frac_total_zeros - zero_total_budget) ** 2
    else:
        penalty_sparse = 0.0

    # --- Total distance ---
    distance = dist_shape + alpha * penalty_bad + beta_sparse * penalty_sparse

    return distance


def loss_distance_q_Real(params, q_true):
    # Parameters used to compute IFC from neural-network outputs
    # params = [d_layers_nm, angles_deg, wavelength_mu, eps_superstrate, eps_substrate]

    # Optimization parameters
    params_opt_max = [1000, 2.0, 50, 50]
    isofreq = partial(rqf.isofreq_minuit_Rp, params_opt=params_opt_max)
    isofreq_sem = partial(rqf.isofreq_Rp, params_opt=params_opt_max)
    ctes = rqf.Ctes_Perm(params)

    # Compute IFC
    q_mode_1 = isofreq_sem(ctes)
    q_real_NN, q_imag_NN = isofreq(ctes, q_mode_1)

    # loss = MSE_IFC(q_true,q_real_NN)
    distance = distance_q(q_true, q_real_NN)
    return distance, q_real_NN


def y_NN_min_oracle(
    ytr_NN,
    positions,
    positions_input,
    q_true,
    input2,
    Nbranches,
    Nfeatures,
    num_ifc,
    mat=["MoO3", "MoO3", "MoO3"],
    subs="SiO2",
    bilayer=True,
):
    """
    Select the branch prediction that minimizes oracle distance to ``q_true``.

    Args:
        ytr_NN: Network predictions with shape ``(1, Nbranches * Nfeatures)``.
        positions: Binary mask indicating parameters predicted by the network.
        positions_input: Binary mask indicating parameters provided via ``input2``.
        q_true: Target IFC (1D array).
        input2: Values for parameters that are not predicted.
        Nbranches: Number of output branches.
        Nfeatures: Number of parameters per branch.
        num_ifc: Number of best branches to return.
        mat, subs, bilayer: Physical system descriptors.

    Returns:
        Tuple containing best-branch parameters, top-k branch parameters, minimum
        distance value, best predicted IFC, top-k predicted IFCs, and associated
        reconstructed physical parameter vectors.
    """
    d_layers = np.ones(3)
    angles = np.ones(2)
    if bilayer:
        if input2 is None:
            input2 = np.zeros(1, dtype=object)
        else:
            input2 = np.insert(input2, 0, 0)
        positions = np.insert(positions, 0, 0)
        positions = np.insert(positions, 2, 0)
        if positions_input[0] == 0:
            input2 = np.insert(input2, 0, 0)
        else:
            input2 = np.insert(input2, 2, 0)

    eps_superstrate = 1.0
    list_distances = []
    q_NN_list = []
    params_list = []

    for i in range(Nbranches):
        a = 0  # current index in ytr_NN
        b = 0  # current index in input2
        d_layers_i = np.copy(d_layers)
        angles_i = np.copy(angles)
        w = None
        wavelength_mu = None

        for j in range(len(positions)):
            if j <= 1:
                if positions[j] == 1:
                    angles_i[j] = ytr_NN[0, i * Nfeatures + a]
                    a += 1
                else:
                    angles_i[j] = input2[b]
                    b += 1
            elif 2 <= j <= 4:
                if positions[j] == 1:
                    d_layers_i[j - 2] = ytr_NN[0, i * Nfeatures + a]
                    a += 1
                else:
                    d_layers_i[j - 2] = input2[b]
                    b += 1
            else:
                if positions[j] == 1:
                    w = np.array([ytr_NN[0, i * Nfeatures + Nfeatures - 1]])
                    wavelength_mu = 1e4 / w
                else:
                    w = np.array([input2[b]])
                    wavelength_mu = 1e4 / w

        params = [d_layers_i, angles_i, wavelength_mu, eps_superstrate, subs, mat]
        distance, q_NN = loss_distance_q_Real(params, q_true)
        params_vec = np.concatenate([angles_i, d_layers_i, w, np.array([distance])])
        params_list.append(params_vec)
        list_distances.append(distance)
        q_NN_list.append(q_NN.reshape(-1, 1))

    params_list = np.stack(params_list, axis=1)
    q_NN_list = np.concatenate(q_NN_list, axis=1)
    list_distances = np.array(list_distances)

    idx_min = np.argmin(list_distances)
    if num_ifc < len(list_distances):
        idx_Nmin = np.argpartition(list_distances, num_ifc)[:num_ifc]
    else:
        idx_Nmin = np.arange(0, min(num_ifc, len(list_distances)))

    angles_thickness_min_distance = ytr_NN[
        0, idx_min * Nfeatures : (idx_min + 1) * Nfeatures
    ]
    q_NN_min_distance = q_NN_list[:, idx_min]
    params_min = params_list[:, idx_min]

    angles_thickness_min_distances = [
        ytr_NN[0, idx * Nfeatures : (idx + 1) * Nfeatures] for idx in idx_Nmin
    ]
    q_NN_list_mins = q_NN_list[:, idx_Nmin]
    params_list_mins = params_list[:, idx_Nmin]

    return (
        angles_thickness_min_distance,
        angles_thickness_min_distances,
        np.min(list_distances),
        q_NN_min_distance,
        q_NN_list_mins,
        params_min,
        params_list_mins,
    )


# =============================================================================
# Oracle distance and loss functions
# =============================================================================


def plot_best(qx, qy, params, carpeta, title=None, d=0, a=0, cota=30):
    """Plot the target IFC and best predicted IFC, then save to disk."""
    a_2 = params[:2]
    d_2 = params[2:5]
    dist = params[5]
    w = params[-1]

    plt.rcParams["figure.figsize"] = (6, 6)
    fig, ax = plt.subplots()
    ax.set(xlim=(-cota, cota), ylim=(-cota, cota))
    ax.scatter(
        qx[1],
        qy[1],
        s=5,
        c="red",
        label=f"{np.round(d_2).astype(int)}     {np.round(a_2).astype(int)}   {np.round(w).astype(int)}   {np.round(dist).astype(int)}",
    )
    ax.scatter(qx[0], qy[0], s=5, c="blue")
    # Remove axis ticks/labels for clean figure export
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tick_params(axis="both", which="major", labelsize=14)
    filename = f"{carpeta}/{title}q_Best.png"
    plt.savefig(filename, bbox_inches="tight", dpi=500)
    # plt.show()


def plot_multiple(
    qx, qy, q_NN, theta_real_NN, params, carpeta, cota=30, num_ifc=5, num_ifcs=1
):
    """Plot the target IFC against the top-k predicted IFC candidates."""
    plt.rcParams["figure.figsize"] = (6, 6)
    fig, ax = plt.subplots()
    ax.set(xlim=(-cota, cota), ylim=(-cota, cota))

    scatter_plots = []
    scatter_colors = []  # Store colors associated with each scatter series

    for i in range(num_ifc):
        q_real_NN = q_NN[:, i]
        scatter = ax.scatter(
            q_real_NN * np.cos(theta_real_NN),
            q_real_NN * np.sin(theta_real_NN),
            s=5,
            c=f"C{i}",
            label=f"{np.round(params[2:5, i]).astype(int)}        {np.round(params[0:2, i]).astype(int)}  {np.round(params[5, i]).astype(int)}   {np.round(params[-1, i]).astype(int)}",
        )
        scatter_plots.append(scatter)
        scatter_colors.append(f"C{i}")  # Capture color of each scatter series

    scatter_original = ax.scatter(qx, qy, s=5, c="blue", label="Desired")

    # Remove axis ticks/labels for clean figure export
    ax.set_xticks([])
    ax.set_yticks([])

    legend_labels = [
        f"{np.round(params[2:5, i]).astype(int)}     {np.round(params[0:2, i]).astype(int)}  {np.round(params[5, i]).astype(int)}  {np.round(params[-1, i], decimals=2)}"
        for i in range(num_ifc)
    ]

    legend_handles = [
        plt.Line2D([], [], color=color, linewidth=0, label=label)
        for color, label in zip(scatter_colors, legend_labels)
    ]

    # legend = ax.legend(handles=legend_handles, title=r'[$t_1$, $t_2$, $t_3$] (nm)     [$\theta_1$, $\theta_2$] ($^{\circ}$) [w] [distancia]', title_fontsize='17', fontsize='10')
    legend = ax.legend(handles=legend_handles, fontsize="10")

    # Ajustar el color del texto de la leyenda al color correspondiente
    for text, color in zip(legend.get_texts(), scatter_colors):
        text.set_color(color)

    # Ajustes adicionales
    legend.get_frame().set_alpha(
        0.0
    )  # Hacer la leyenda completamente transparente Texto "k0"
    plt.tick_params(axis="both", which="major", labelsize=17)

    filename = f"{carpeta}/{num_ifcs}q_several_best.png"
    plt.savefig(filename, bbox_inches="tight", dpi=500)
    # plt.show()
    pass


# =============================================================================
# Hand-drrawn functions
# =============================================================================


def rotate_array(arr, n):
    n = n % len(arr)  # Ensure n remains in the range [0, len(arr))
    return np.concatenate((arr[n:], arr[:n]))


def normalize_angle(angle):
    angle[angle >= 3 * np.pi / 2] = angle[angle >= 3 * np.pi / 2] - 2 * np.pi
    angle[angle < -np.pi / 2] = angle[angle < -np.pi / 2] + 2 * np.pi
    return angle


def canalization(k):
    theta = np.linspace(0, np.pi, 500)
    q = abs(k / (np.sin(theta) - np.cos(theta) * 0.76))
    q[q > 50] = 0

    q = np.reshape(
        np.concatenate([np.reshape(q, (1, 500)), np.reshape(q, (1, 500))], axis=1),
        (1000,),
    )
    thetas = normalize_angle(theta + np.pi)
    thetas_2 = np.reshape(
        np.concatenate(
            [np.reshape(theta, (1, 500)), np.reshape(thetas, (1, 500))], axis=1
        ),
        (1000,),
    )

    indices_sorted = np.argsort(thetas_2)
    q = q[indices_sorted]
    thetas_2 = thetas_2[indices_sorted]

    return q, thetas_2
