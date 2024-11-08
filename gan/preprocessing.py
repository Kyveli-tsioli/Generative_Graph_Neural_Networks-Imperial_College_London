"""
This file contains the functions to preprocess the data before training the GAN.

It was adapted from the following source:
https://github.com/basiralab/AGSR-Net

A number of functions were removed and added, so the code here bears little resemblance to the original implementation.
"""

import networkx as nx
import numpy as np
import torch

from set_seed import set_seed

set_seed(42)


def init_x(A, dim, device, method='eye'):
    """
    Initialise the X matrix for the given adjacency matrix A. Includes options for eye, random, and topology-based
    initialisation.

    After experimenting with the different initialisation methods, it was found that the topology-based method
    consistently produced the best results. This method calculates the average of the centrality, PageRank, betweenness,
    closeness, degree, and clustering coefficient values for each node in the graph, and uses this average to initialise
    the X matrix.
    """
    if method == 'eye':
        return torch.eye(dim, dtype=torch.float32, device=device)
    elif method == 'random':
        return torch.rand(dim, dim, dtype=torch.float32, device=device)
    elif method == 'topology':
        centrality = torch.tensor(list(nx.eigenvector_centrality_numpy(nx.from_numpy_array(A)).values()),
                                  device=device, dtype=torch.float32)
        pagerank = torch.tensor(list(nx.pagerank(nx.from_numpy_array(A)).values()), device=device,
                                dtype=torch.float32)
        betweenness = torch.tensor(list(nx.betweenness_centrality(nx.from_numpy_array(A)).values()),
                                   device=device, dtype=torch.float32)
        closeness = torch.tensor(list(nx.closeness_centrality(nx.from_numpy_array(A)).values()),
                                 device=device, dtype=torch.float32)
        degree = torch.tensor(list(nx.degree_centrality(nx.from_numpy_array(A)).values()), device=device,
                              dtype=torch.float32)
        clustering = torch.tensor(list(nx.clustering(nx.from_numpy_array(A)).values()), device=device,
                                  dtype=torch.float32)
        # average those values
        avg = (centrality + pagerank + betweenness + closeness + degree + clustering) / 6
        return torch.diag(avg)


def preprocess_data(subjects_adj, args):
    """
    Preprocess the data for the given subjects. This includes normalising the adjacency matrices and initialising the
    X matrices. Importantly, the data is converted from numpy arrays to PyTorch tensors.
    """
    lr_adj = []
    lr_x = []
    for lr in subjects_adj:
        lr_A = torch.from_numpy(lr).type(torch.FloatTensor).to(args.device)
        lr_X = init_x(lr, args.lr_dim, args.device, method=args.init_x_method)
        lr_A_norm = args.normalisation_function(nx.from_numpy_array(lr), lr_A)
        lr_adj.append(lr_A_norm)
        lr_x.append(lr_X)

    return torch.stack(lr_adj), torch.stack(lr_x)


def pad_HR_adj(label, split):
    """
    Pad the given label to the given split value.
    """
    label = np.pad(label, ((split, split), (split, split)), mode="constant")
    np.fill_diagonal(label, 1)
    return label


def unpad_HR_adj(label, split):
    """
    Unpad the given label to the given split value.
    """
    return label[split:-split, split:-split]


# The different normalisation functions we tried out. After experimenting with the different normalisation methods,
# it was found that the degree normalisation method consistently produced the best results. This method divides the
# adjacency matrix by the degree of each node in the graph.

def degree_normalisation(G, adjacency):
    """
    Normalise the given adjacency matrix using the degree normalisation method.
    """
    epsilon = 1e-6
    degree_matrix = torch.diag(1. / (torch.sum(adjacency, dim=1) + epsilon))
    return torch.mm(degree_matrix, adjacency)


def pagerank_normalisation(G, adjacency):
    """
    Normalise the given adjacency matrix using the PageRank normalisation method.
    """
    page_ranks = nx.pagerank(G)
    return torch.mm(torch.diag(torch.tensor(list(page_ranks.values()), dtype=torch.float, device=adjacency.device)),
                    adjacency)


def betweenness_normalisation(G, adjacency):
    """
    Normalise the given adjacency matrix using the betweenness centrality normalisation method.
    """
    betweenness = nx.betweenness_centrality(G)
    return torch.mm(torch.diag(torch.tensor(list(betweenness.values()), dtype=torch.float, device=adjacency.device)),
                    adjacency)


def clustering_coefficient_normalisation(G, adjacency):
    """
    Normalise the given adjacency matrix using the clustering coefficient normalisation method.
    """
    clustering = nx.clustering(G)
    return torch.mm(torch.diag(torch.tensor(list(clustering.values()), dtype=torch.float, device=adjacency.device)),
                    adjacency)
