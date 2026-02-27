# -*- coding: utf-8 -*-
"""
===========================================================
Models
===========================================================
Deep Learning models for predicting parameters in twisted multilayer optical systems.
Defines Keras/TensorFlow models for regression tasks using both spectral (q) and scalar (input2) features.

Model architecture:
- Input 0: 1D array of q values (shape: [q_size, 1]) processed through Conv1D and pooling layers.
    - q_size: Number of q points (e.g., 500) for the isofrequency contour.
- Input 2: Optional vector of scalar features (shape: [shape_input_2]) processed through a Dense layer.
    - shape_input_2: Number of scalar input features (e.g., thickness, angle, etc.).

- Latent layer: Dense layer combining spectral and scalar features.
- hidden_neurons_branch: Number of neurons in the hidden layer of each output branch.
- Nfeatures: Number of parameters to predict per branch.
- Nbranches: Number of output branches (parallel predictions).

- Output branches: Multiple parallel branches (Nbranches) each with a hidden layer and an output layer predicting Nfeatures parameters.
The model supports two inputs: a 1D q array and an optional vector of scalar features.

Author: [Lucia F. Alvarez-Tomillo]
Date: [07/11/2025]
"""

########## IMPORT PACKAGES ##########

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.layers import Dense, concatenate

########## KERAS MODELS ##########


def model_twistoptics(
    q_size, shape_input_2, hidden_neurons_branch, Nfeatures, Nbranches
):
    """
    Build a Keras model for twisted multilayer optics regression.

    Parameters
    ----------
    q_size : int
        Number of q points (input0 shape).
    shape_input_2 : int or None
        Number of scalar input features (input2 shape). If None, only q is used.
    hidden_neurons_branch : int
        Number of neurons in the hidden layer of each output branch.
    Nfeatures : int
        Number of parameters to predict per branch.
    Nbranches : int
        Number of output branches (parallel predictions).

    Returns
    -------
    model : tf.keras.Model
        Compiled Keras model.
    """
    # Input 0: q values (spectral input)
    kernel = (50,)
    input_shape_conv1d = (q_size, 1)
    input0 = Input(shape=input_shape_conv1d, name="input0")
    x = layers.Conv1D(32, kernel_size=kernel, activation="relu")(input0)
    x = layers.MaxPooling1D(pool_size=8)(x)
    x = layers.Conv1D(16, (8,), activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.Conv1D(16, (4,), activation="relu")(x)
    y = layers.Flatten()(x)

    # Input 2: scalar features (e.g., thickness, angle, etc.)
    if shape_input_2:
        input2 = Input(shape=shape_input_2, name="input2")
        concat2 = Dense(10, activation="relu")(input2)
        # Concatenate spectral and scalar features
        t = concatenate([y, concat2])
        t = Dense(50, activation="relu")(t)
    else:
        t = Dense(50, activation="relu")(y)

    # Latent layer before output branches
    kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1 / np.sqrt(32))
    bias_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0)
    pre_output = Dense(
        Nfeatures,
        activation="relu",
        kernel_initializer=kernel_init,
        bias_initializer=bias_init,
    )(t)

    # Create output branches
    list_branches_1 = []
    list_branches_2 = []
    for j in range(Nbranches):
        # First dense layer in branch
        list_branches_1.append(
            Dense(
                hidden_neurons_branch,
                activation="sigmoid",
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=1 / np.sqrt(Nfeatures)
                ),
                bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1),
            )
        )
        # Output layer in branch
        list_branches_2.append(
            Dense(
                Nfeatures,
                activation="sigmoid",
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0, stddev=1 / np.sqrt(hidden_neurons_branch)
                ),
                bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1),
            )
        )

    # Connect branches to the latent layer
    outputs_concatenated = []
    for j in range(Nbranches):
        branch_output_1 = list_branches_1[j](pre_output)
        branch_output_2 = list_branches_2[j](branch_output_1)
        outputs_concatenated.append(branch_output_2)
    outputs_concatenated = tf.keras.backend.concatenate(outputs_concatenated)

    # Build model with or without input2
    if shape_input_2:
        model = Model(inputs=[input0, input2], outputs=outputs_concatenated)
    else:
        model = Model(inputs=input0, outputs=outputs_concatenated)
    return model
