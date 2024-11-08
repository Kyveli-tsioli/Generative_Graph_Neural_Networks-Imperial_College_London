"""
This file contains the implementation of the Graph Spectral Regularization (GSR) layer.

It was adapted from the following source:
https://github.com/basiralab/AGSR-Net

The main change made to the original implementation was to change the shape of the weights matrix to avoid unnecessary
padding.
"""

import torch
import torch.nn as nn

from gan.initializations import weight_variable_glorot
from set_seed import set_seed

set_seed(42)


class GSRLayer(nn.Module):

    def __init__(self, lr_dim, hr_dim):
        super(GSRLayer, self).__init__()

        self.weights = torch.from_numpy(
            weight_variable_glorot(lr_dim * 2, hr_dim)).type(torch.FloatTensor)
        self.weights = torch.nn.Parameter(
            data=self.weights, requires_grad=True)

    def forward(self, A, X):
        with torch.autograd.set_detect_anomaly(True):
            lr = A
            lr_dim = lr.shape[0]
            f = X
            eig_val_lr, U_lr = torch.linalg.eigh(lr, UPLO='U')

            # U_lr = torch.abs(U_lr)
            eye_mat = torch.eye(lr_dim).type(torch.FloatTensor)
            s_d = torch.cat((eye_mat, eye_mat), 0).to(X.device)

            a = torch.matmul(self.weights, s_d)
            b = torch.matmul(a, torch.t(U_lr))
            f_d = torch.matmul(b, f)
            f_d = torch.abs(f_d)
            f_d = f_d.fill_diagonal_(1)
            adj = f_d

            X = torch.mm(adj, adj.t())
            X = (X + X.t()) / 2
            X = X.fill_diagonal_(1)
        return adj, torch.abs(X)
