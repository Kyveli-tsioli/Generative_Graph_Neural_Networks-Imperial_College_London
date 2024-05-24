"""
utils.py - Utility Functions

Description:
This module provides functions that may be useful for all the models used in this project, mostly related with:
pre-processing, post-processing and evaluation
"""

# Imports
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from sklearn.model_selection import KFold

from Evaluator import evaluate
from MatrixVectorizer import MatrixVectorizer
from constants import *
from set_seed import set_seed

set_seed(42)


def load_csv_files(return_matrix=False, include_diagonal=False):
    """
    Load CSV files and perform pre-processing.

    Parameters:
    - return_matrix (bool): If True, applies anti-vectorization to return matrices.
    - include_diagonal (bool): If True and return_matrix is True, includes diagonal elements in anti-vectorization.

    Returns:
    - Tuple of NumPy arrays: Contains loaded data (lr_train, hr_train, lr_test).
    """

    hr_train_data = np.genfromtxt("data/hr_train.csv", delimiter=",", skip_header=1)
    lr_train_data = np.genfromtxt("data/lr_train.csv", delimiter=",", skip_header=1)
    lr_test_data = np.genfromtxt("data/lr_test.csv", delimiter=",", skip_header=1)

    # Pre-processing of the values

    np.nan_to_num(hr_train_data, copy=False)
    np.nan_to_num(lr_train_data, copy=False)
    np.nan_to_num(lr_test_data, copy=False)

    hr_train_data = np.maximum(hr_train_data, 0)
    lr_train_data = np.maximum(lr_train_data, 0)
    lr_test_data = np.maximum(lr_test_data, 0)

    if return_matrix:
        # Apply anti-vectorization
        hr_train_matrices = np.empty(
            (hr_train_data.shape[0], HR_MATRIX_SIZE, HR_MATRIX_SIZE)
        )
        for i, sample in enumerate(hr_train_data):
            hr_train_matrices[i] = MatrixVectorizer.anti_vectorize(
                sample, HR_MATRIX_SIZE, include_diagonal
            )

        lr_train_matrices = np.empty(
            (lr_train_data.shape[0], LR_MATRIX_SIZE, LR_MATRIX_SIZE)
        )
        for i, sample in enumerate(lr_train_data):
            lr_train_matrices[i] = MatrixVectorizer.anti_vectorize(
                sample, LR_MATRIX_SIZE, include_diagonal
            )

        lr_test_matrices = np.empty(
            (lr_test_data.shape[0], LR_MATRIX_SIZE, LR_MATRIX_SIZE)
        )
        for i, sample in enumerate(lr_test_data):
            lr_test_matrices[i] = MatrixVectorizer.anti_vectorize(
                sample, LR_MATRIX_SIZE, include_diagonal
            )

        return lr_train_matrices, hr_train_matrices, lr_test_matrices

    return lr_train_data, hr_train_data, lr_test_data


def save_csv_prediction(
        predictions, pred_matrix=False, include_diagonal=False, logs=False, file_name=None
):
    """
    Save predictions in CSV format.

    Parameters:
    - predictions: Numpy array containing the predictions of test set.
    - pred_matrix (bool): If True, assumes predictions is an adjacency matrix and applies vectorization.
    - include_diagonal (bool): If True and input_matrix is True, includes diagonal elements in vectorization.
    - logs (bool): If True, prints the shape of the vectorized matrix.
    """

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    if pred_matrix:
        # Apply vectorization
        df_vector = np.empty((predictions.shape[0], HR_ARRAY_SIZE))
        for i, prediction in enumerate(predictions):
            df_vector[i] = MatrixVectorizer.vectorize(prediction, include_diagonal)

        predictions = df_vector
        if logs:
            print(predictions.shape)

    # Post-processing
    predictions = np.maximum(predictions, 0)
    array_d1 = predictions.flatten()
    array_indexes = np.vstack((np.arange(1, len(array_d1) + 1), array_d1)).T

    if file_name is None:
        file_name = "submissions/" + time.strftime("%Y%m%d-%H%M%S") + "_submission.csv"
    else:
        file_name = "submissions/" + file_name + ".csv"

    np.savetxt(
        file_name,
        array_indexes,
        delimiter=",",
        fmt="%i,%f",
        header="ID,Predicted",
        comments="",
    )
    if logs:
        print(f"CSV file '{file_name}' created.")


def three_fold_cross_validation(
        model_init,
        X,
        Y,
        label_vector=True,
        random_state=None,
        prediction_vector=False,
        include_diagonal=False,
        verbose=False,
):
    """
    Perform three-fold cross-validation on a given model.

    Parameters:
    - model_init: Function that returns an instance of the model to be evaluated.
    - X: The input features.
    - Y: The target labels
    - label_vector: Whether the target labels are vectorized or not.
    - prediction_vector: Whether the predictions are vectorized or not.
    - random_state: Seed for reproducibility in cross-validation.
    - include_diagonal: Include diagonal elements during vectorization if input_matrix is True.
    - verbose: Print logging information.
    - device: Device to use for training and evaluation.

    Returns:
    - scores: List of evaluation scores for each fold.
    """

    kf = KFold(n_splits=3, shuffle=True, random_state=random_state)

    scores = []
    training_time = 0

    process = psutil.Process()

    for index, (train_index, val_index) in enumerate(kf.split(X)):
        model = model_init()
        start_time = time.time()
        model.fit(X[train_index], Y[train_index], verbose=verbose)
        end_time = time.time()
        training_time += end_time - start_time

        Y_prediction = model.predict(X[val_index])

        args = {"include_diagonal": include_diagonal, "verbose": verbose}

        if label_vector:
            truths_vectors = Y[val_index]
            assert len(truths_vectors.shape) == 2
            if isinstance(truths_vectors, torch.Tensor):
                truths_vectors = truths_vectors.cpu().numpy()
            args["truths_vectors"] = truths_vectors
        else:
            truths_matrices = Y[val_index]
            assert len(truths_matrices.shape) == 3
            if isinstance(truths_matrices, torch.Tensor):
                truths_matrices = truths_matrices.cpu().numpy()
            args["truths_matrices"] = truths_matrices

        if prediction_vector:
            args["predictions_vectors"] = Y_prediction
        else:
            args["predictions_matrices"] = Y_prediction.cpu().numpy()

        evaluations = evaluate(**args)

        scores.append(evaluations)

        save_csv_prediction(Y_prediction, pred_matrix=True, include_diagonal=include_diagonal, logs=verbose,
                            file_name=f"predictions_fold_{index}")

    ram_usage = process.memory_info().rss / (1024 ** 2)  # in MBs

    if verbose:
        print(f"Training time: {training_time} seconds")
        print(f"RAM usage: {ram_usage} MB")

    return scores


def plot_evaluations(scores):
    """
    Plot performance evaluations for each fold and their average.

    Parameters:
    - scores (list): List containing three sub-lists, each representing the evaluation scores for a fold.
    """

    assert len(scores) == 3
    if len(scores[0]) == 8:
        labels = ["MAE", "PCC", "JSD", "BC", "EC", "PC", "DC", "CL"]
        colors = [
            "indianred",
            "orange",
            "limegreen",
            "aquamarine",
            "royalblue",
            "hotpink",
            "indigo",
            "darkturquoise",
        ]
    else:
        labels = ["MAE", "PCC", "JSD", "BC", "EC", "PC", "DC", "CL", "FID"]
        colors = [
            "indianred",
            "orange",
            "limegreen",
            "aquamarine",
            "royalblue",
            "hotpink",
            "indigo",
            "darkturquoise",
            "darkorange",
        ]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.suptitle("Performance Measures for Each Fold and Average", fontsize=16)

    for i, fold_evaluation in enumerate(scores):
        row = i // 2
        col = i % 2

        axes[row, col].bar(range(len(fold_evaluation)), fold_evaluation, color=colors)
        axes[row, col].set_title("Fold " + str(i + 1))
        # set the top of the y-axis to the maximum value found
        axes[row, col].set_ylim(0, 1.1 * max(fold_evaluation))
        axes[row, col].set_xticks(np.arange(len(labels)))
        axes[row, col].set_xticklabels(labels)

    average_folds = np.mean(np.array(scores), axis=0)
    std_folds = np.std(np.array(scores), axis=0)
    axes[1, 1].bar(
        range(len(average_folds)), average_folds, color=colors, yerr=std_folds
    )
    axes[1, 1].set_title("Average Across Folds")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xticks(np.arange(len(labels)))
    axes[1, 1].set_xticklabels(labels)

    plt.show()
    plt.savefig("performance_measures.png")
