"""
This file contains the GUS and Discriminator classes, which are used to define the neural network models for the
GUS-GAN model.

It was adapted from the following source:
https://github.com/basiralab/AGSR-Net

The main change made to the original implementation include:
- changing the u_net implementation to use the GraphUNet class from the PyTorch Geometric library
- changing the convolutional layer to use the SSGConv class from the PyTorch Geometric library instead of the basic
    graph convolutional layer
- changing the normalisation function to be a parameter of the Args class which allows for the normalisation function to
    be easily adjusted (degree, pagerank, etc.)
- changing the initialisation method to be a parameter of the Args class which allows for the initialisation method to \
    be easily adjusted (topology, eye, random)
"""

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch_geometric.utils as gutils

from gan.layers import GSRLayer
from set_seed import set_seed

set_seed(42)


class GUS(nn.Module):
    """
    The GUS model (GRSLayer-U-Net-SSGConv) used for the GAN model. Based on the AGSR-Net model.
    """

    def __init__(self, ks, args):
        super(GUS, self).__init__()

        self.lr_dim = args.lr_dim
        self.hr_dim = args.hr_dim
        self.hidden_dim = args.hidden_dim
        self.layer = GSRLayer(self.lr_dim, self.hr_dim)
        self.u_net = gnn.GraphUNet(self.lr_dim, self.hidden_dim, self.hr_dim, depth=len(ks), pool_ratios=ks)
        self.gc1 = gnn.SSGConv(self.hr_dim, self.hidden_dim, args.alpha, args.k)
        self.gc2 = gnn.SSGConv(self.hidden_dim, self.hr_dim, args.alpha, args.k)

        self.device = args.device
        self.normalisation_function = args.normalisation_function
        self.args = args

    def forward(self, lr_A, X):
        """
        Forward pass of the model. This is used to predict the output for the given input data.
        The forward pass includes:
        - the U-Net layer
        - the GSR layer
        - the graph convolutional layers
        - final averaging and filling of the diagonal
        """
        with torch.autograd.set_detect_anomaly(True):
            edge_index = gutils.dense_to_sparse(lr_A)[0].to(self.device)
            net_outs = self.u_net(X, edge_index)

            A_hat, X_hat = self.layer(lr_A, net_outs)

            edge_index, edge_weights = gutils.dense_to_sparse(A_hat)

            X_hat = self.gc1(X_hat, edge_index, edge_weights)
            X_hat = self.gc2(X_hat, edge_index, edge_weights)

            X_hat = (X_hat + X_hat.t()) / 2
            X_hat = X_hat.fill_diagonal_(1)

        # return torch.abs(z), self.net_outs, self.start_gcn_outs
        return torch.abs(X_hat)

    def fit(self, X, Y, verbose=False):
        """
        Train the model using the given data. This is needed for the Cross-Validation implementation.
        """
        from gan.train import train
        best_model = train(self, X, Y, self.args, verbose=verbose)
        self.load_state_dict(best_model.state_dict())
        return self

    def predict(self, X):
        """
        Predict the output for the given input data.
        """
        self.eval()
        with torch.no_grad():
            # this is a batch, we need to iterate over it
            pred = []
            for (lr_A, lr_X) in X:
                p = self(lr_A, lr_X)
                pred.append(p)
        self.train()
        # create a tensor from the list of tensors
        pred = torch.stack(pred)
        return pred


class Discriminator(nn.Module):
    """
    The Discriminator model used for regularising the GUS model during training.
    """

    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(args.hr_dim, args.hr_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hr_dim, args.hr_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hr_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        output = self.net(inputs)
        return torch.abs(output)


def gaussian_noise_layer(input_layer, args):
    """
    Gaussian noise layer used for creating the noise for the GAN training.
    """
    z = torch.empty_like(input_layer)
    noise = z.normal_(mean=args.mean_gaussian, std=args.std_gaussian)
    z = torch.abs(input_layer + noise)

    z = (z + z.t()) / 2
    z = z.fill_diagonal_(1)
    return z
