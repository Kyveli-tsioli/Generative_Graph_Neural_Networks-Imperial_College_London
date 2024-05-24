# Adapted from: https://github.com/basiralab/DGL/blob/main/Project/evaluation_measures.py

import networkx as nx
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from MatrixVectorizer import MatrixVectorizer
from constants import *
from set_seed import set_seed

set_seed(42)


def evaluate(
        truths_vectors=None,
        truths_matrices=None,
        predictions_matrices=None,
        predictions_vectors=None,
        include_diagonal=False,
        verbose=False,
        include_fid=False,
):
    """
    Evaluate the performance of centrality prediction on graph data.

    Parameters:
    - truths_vectors (numpy array): Ground truth in vectorized form. If this parameter is not provided, truths_matrices
    is required.
    - truths_matrices (numpy array): Ground truth in matrix form. If this parameter is not provided, truths_vectors is
    required.
    - predictions_matrices (numpy array, optional): Predicted adjacency matrices, if this parameter is not provided,
    predictions_vectors is required.
    - predictions_vectors (numpy array, optional): Predicted centrality values for nodes, if this parameter is not
    provided, predictions_matrices is required.
    - include_diagonal (bool, optional): Include diagonal elements in computations.
    - verbose (bool, optional): Print intermediate results if True.
    - include_fid (bool, optional): Include Frechet Inception Distance (FID) computation if True.
    - device (str, optional): Device to use for computations.

    Returns:
    - List containing [MAE, PCC, Jensen-Shannon Distance, Avg MAE Betweenness Centrality, Avg MAE Eigenvector
    Centrality, Avg MAE PageRank Centrality, Avg MAE Degree Centrality, Avg MAE Clustering Coefficient].
    If include_fid is True, FID is also included in the list.
    """

    # Check on optional inputs
    assert predictions_matrices is not None or predictions_vectors is not None
    assert truths_matrices is not None or truths_vectors is not None

    if predictions_matrices is None:
        # Apply anti-vectorization
        predictions_matrices = np.empty(
            (predictions_vectors.shape[0], HR_MATRIX_SIZE, HR_MATRIX_SIZE)
        )
        for i, prediction in enumerate(predictions_vectors):
            predictions_matrices[i] = MatrixVectorizer.anti_vectorize(
                prediction, HR_MATRIX_SIZE, include_diagonal
            )
    else:
        # Apply vectorization
        predictions_vectors = np.empty((predictions_matrices.shape[0], HR_ARRAY_SIZE))
        for i, prediction in enumerate(predictions_matrices):
            predictions_vectors[i] = MatrixVectorizer.vectorize(
                prediction, include_diagonal
            )

    # Apply anti-vectorization on truth
    if truths_matrices is None:
        truths_matrices = np.empty(
            (truths_vectors.shape[0], HR_MATRIX_SIZE, HR_MATRIX_SIZE)
        )
        for i, truth in enumerate(truths_vectors):
            truths_matrices[i] = MatrixVectorizer.anti_vectorize(
                truth, HR_MATRIX_SIZE, include_diagonal
            )
    else:
        truths_vectors = np.empty((truths_matrices.shape[0], HR_ARRAY_SIZE))
        for i, truth in enumerate(truths_matrices):
            truths_vectors[i] = MatrixVectorizer.vectorize(truth, include_diagonal)

    num_test_samples = predictions_matrices.shape[0]

    # post-processing on predictions
    predictions_matrices = np.maximum(predictions_matrices, 0)
    predictions_vectors = np.maximum(predictions_vectors, 0)

    # Initialize lists to store MAEs for each centrality measure
    mae_bc = []
    mae_ec = []
    mae_pc = []
    mae_dc = []
    mae_cl = []

    # Iterate over each test sample
    for i in range(num_test_samples):

        if verbose:
            print(i)

        # Convert adjacency matrices to NetworkX graphs
        if isinstance(predictions_matrices[i], torch.Tensor):
            predictions_matrix = predictions_matrices[i].cpu().numpy()
        else:
            predictions_matrix = predictions_matrices[i]
        if isinstance(truths_matrices[i], torch.Tensor):
            truths_matrix = truths_matrices[i].cpu().numpy()
        else:
            truths_matrix = truths_matrices[i]
        pred_graph = nx.from_numpy_array(predictions_matrix)
        gt_graph = nx.from_numpy_array(truths_matrix)

        # Compute centrality measures
        pred_bc = nx.betweenness_centrality(pred_graph, weight="weight")
        pred_ec = nx.eigenvector_centrality(pred_graph, weight="weight")
        pred_pc = nx.pagerank(pred_graph, weight="weight")
        pred_dc = nx.degree_centrality(pred_graph)
        pred_cl = nx.clustering(pred_graph, weight="weight")

        gt_bc = nx.betweenness_centrality(gt_graph, weight="weight")
        gt_ec = nx.eigenvector_centrality(gt_graph, weight="weight")
        gt_pc = nx.pagerank(gt_graph, weight="weight")
        gt_dc = nx.degree_centrality(gt_graph)
        gt_cl = nx.clustering(gt_graph, weight="weight")

        # Convert centrality dictionaries to lists
        pred_bc_values = list(pred_bc.values())
        pred_ec_values = list(pred_ec.values())
        pred_pc_values = list(pred_pc.values())
        pred_dc_values = list(pred_dc.values())
        pred_cl_values = list(pred_cl.values())

        gt_bc_values = list(gt_bc.values())
        gt_ec_values = list(gt_ec.values())
        gt_pc_values = list(gt_pc.values())
        gt_dc_values = list(gt_dc.values())
        gt_cl_values = list(gt_cl.values())

        # Compute MAEs
        mae_bc.append(mean_absolute_error(pred_bc_values, gt_bc_values))
        mae_ec.append(mean_absolute_error(pred_ec_values, gt_ec_values))
        mae_pc.append(mean_absolute_error(pred_pc_values, gt_pc_values))
        mae_dc.append(mean_absolute_error(pred_dc_values, gt_dc_values))
        mae_cl.append(mean_absolute_error(pred_cl_values, gt_cl_values))

    # Compute average MAEs
    avg_mae_bc = sum(mae_bc) / len(mae_bc)
    avg_mae_ec = sum(mae_ec) / len(mae_ec)
    avg_mae_pc = sum(mae_pc) / len(mae_pc)
    avg_mae_dc = sum(mae_dc) / len(mae_dc)
    avg_mae_cl = sum(mae_cl) / len(mae_cl)

    # vectorize and flatten
    pred_1d = predictions_vectors.flatten()
    gt_1d = truths_vectors.flatten()

    mae = mean_absolute_error(pred_1d, gt_1d)
    pcc = pearsonr(pred_1d, gt_1d)[0]
    js_dis = jensenshannon(pred_1d, gt_1d)

    if include_fid:
        # FID - Expensive computation
        if isinstance(predictions_vectors, np.ndarray):
            predictions_tensors = torch.from_numpy(predictions_vectors)
        else:
            predictions_tensors = predictions_vectors
        if isinstance(truths_vectors, np.ndarray):
            truths_tensors = torch.from_numpy(truths_vectors)
        else:
            truths_tensors = truths_vectors
        mu_real = torch.mean(truths_tensors, dim=0)
        cov_real = torch.cov(truths_tensors.t())
        mu_gen = torch.mean(predictions_tensors, dim=0)
        cov_gen = torch.cov(predictions_tensors.t())
        fid = torch.sqrt(
            torch.norm(torch.sub(mu_real, mu_gen)) ** 2
            + torch.trace(
                cov_real + cov_gen - 2 * torch.sqrt(torch.matmul(cov_real, cov_gen))
            )
        ).cpu().numpy()

    if verbose:
        print("MAE: ", mae)
        print("PCC: ", pcc)
        print("Jensen-Shannon Distance: ", js_dis)
        print("Average MAE betweenness centrality:", avg_mae_bc)
        print("Average MAE eigenvector centrality:", avg_mae_ec)
        print("Average MAE PageRank centrality:", avg_mae_pc)
        print("Average MAE degree centrality:", avg_mae_dc)
        print("Average MAE clustering coefficient:", avg_mae_cl)
        if include_fid:
            print("Frechet Inception Distance:", fid)
    if include_fid:
        return [mae, pcc, js_dis, avg_mae_bc, avg_mae_ec, avg_mae_pc, avg_mae_dc, avg_mae_cl, fid]
    else:
        return [mae, pcc, js_dis, avg_mae_bc, avg_mae_ec, avg_mae_pc, avg_mae_dc, avg_mae_cl]
