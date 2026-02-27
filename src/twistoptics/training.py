
# -*- coding: utf-8 -*-
"""
===========================================================
Training
===========================================================
Training utilities for multi-branch TwistOptics models.

directory: Required to specify the full path where trained models will be saved
database: Could be replaced by maximum and minimum values from the database
ntrain: Depends on database size
nvalidation: Depends on database size
positions: Most important for your training. If you don't want to fix the parameter in the training, set the corresponding value to 1.
    If you want to fix it, set it to 0 and specify the value in input2.
    Bilayers: [θ1, t1, t2, w]
    Trilayers: [θ1, θ2, t1, t2, t3, w]
max_nbranches: Maximum number of branches (default: 10)
min_branch: Minimum number of branches (default: 1)
resume_training: To continue an interrupted training, specify the number of branches from which to resume.
    The model with that number of branches must exist in the specified directory.
resume_nbranches: The branch from which training will resume. The model with that number of branches must exist in the specified directory.

Author: [Lucia F. Alvarez-Tomillo]
Date: [07/11/2025]
"""

import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf

import twistoptics.models as mtlm
import twistoptics.utils as auxf


def train_models(
    directory: str = "./Models_Trained_bilayers_w_fixed/",
    database: str = "./Database_bilayers_MoO3_MoO3_MoO3",
    ntrain: int = 20000,
    nvalidation: int = 5000,
    positions: Optional[List[int]] = None,
    max_nbranches: int = 10,
    min_branch: int = 1,
    resume_training: bool = False,
    resume_nbranches: int = 2,
) -> None:
    """Train one or more TwistOptics regression models.

    This function preserves the original branch-pruning training workflow used in the
    research code while exposing the main configuration points as function arguments.
    Models, histories, and activity summaries are written under ``directory``.
    """
    positions = positions or [1, 1, 1, 0]
    directory = str(Path(directory))
    database = str(Path(database))
    Path(directory).mkdir(parents=True, exist_ok=True)

    list_nbranches = [max_nbranches - i for i in range(max_nbranches)]

    if len(positions) == 4:
        ref_all = np.array([1, 1, 1, 1])
    else:
        ref_all = np.array([1, 1, 1, 1, 1, 1])
    nfeatures = sum(positions)
    shape_input_2 = int(np.sum(ref_all - np.array(positions)))

    xtr, xva, ytr_single, yva_single, _ = auxf.prepare_data(
        ntrain, nvalidation, positions, database
    )

    early_stopping = True
    q_size = 500
    hidden_neurons_branch = 25
    epochs = 1000 if early_stopping else 300
    minib_size = 125

    seed_list = auxf.load_seed_list()
    num_simulations = 1 #change for training multiple seeds
    list_simulations = [i for i in range(num_simulations)]

    # Loop over seeds
    for i in list_simulations:
        print("SEED = ", seed_list[i])
        auxf.reset_random_seeds(seed_list[i])

        if resume_training:
            list_nbranches = [resume_nbranches - j for j in range(resume_nbranches)]
            copy_list = list(list_nbranches)
            nbranches = copy_list.pop(0)
            model = tf.keras.models.load_model(
                f"{directory}/Model_{i}seed_{nbranches + 1}branches/Model_{i}seed_{nbranches + 1}branches.h5",
                custom_objects={
                    "NBsym_loss": auxf.symmetry_loss_Nb(nbranches, nfeatures)
                },
                compile=False,
            )
            w = model.get_weights()
            ytr = auxf.expand_outputs_for_branches(ytr_single, nbranches + 1)
            y_nn = model.predict(xtr)
            auxf.remove_min_branch(w, ytr, y_nn, nbranches + 1, nfeatures)
        else:
            # Training new model from scratch
            copy_list = list(list_nbranches)
            nbranches = copy_list.pop(0)

        # Loop over branches
        while nbranches >= min_branch:
            ytr = auxf.expand_outputs_for_branches(ytr_single, nbranches)
            yva = auxf.expand_outputs_for_branches(yva_single, nbranches)
            model = mtlm.model_twistoptics(
                q_size, shape_input_2, hidden_neurons_branch, nfeatures, nbranches
            )
            model.compile(
                optimizer="Adam", loss=auxf.symmetry_loss_Nb(nbranches, nfeatures)
            )
            if nbranches < max_nbranches:
                model.set_weights(w)

            file_name = f"Model_{i}seed_{nbranches}branches"
            folder = Path(directory) / file_name
            folder.mkdir(parents=True, exist_ok=True)
            start_time = time.time()
            r = model.fit(
                xtr,
                ytr,
                batch_size=minib_size,
                epochs=300 if nbranches < max_nbranches else epochs,
                verbose=1,
                validation_data=(xva, yva),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        min_delta=1e-6,
                        patience=50,
                        mode="min",
                        restore_best_weights=True,
                    )
                ]
                if early_stopping
                else [],
            )
            print(f"---Training completed in {time.time() - start_time} seconds ---")

            model.save(folder / f"{file_name}.h5", save_format="h5")
            auxf.save_history(r, folder / "history_loss.txt")
            w = model.get_weights()
            y_nn = model.predict(xtr)
            activity = auxf.remove_min_branch(w, ytr, y_nn, nbranches, nfeatures)
            (folder / "activity.txt").write_text(str(activity))
            (folder / "hyperparameters.txt").write_text(
                f"n_train={ntrain}\n"
                f"n_val={nvalidation}\n"
                f"epochs={epochs}\n"
                f"mbs={minib_size}\n"
                f"loss=sym_loss_Nb\noptimizer=Adam"
            )

            nbranches = copy_list.pop(0)


def main() -> None:
    """CLI-style entry point used by ``scripts/train.py``."""
    train_models()


if __name__ == "__main__":
    main()
