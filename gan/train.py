"""
This file contains the training and testing functions for the GAN model.

It was adapted from the following source:
https://github.com/basiralab/AGSR-Net

The main changes include:
- introducing early stopping to the training process
- changing the loss function by removing regularisation terms from the U-Net and the MSE between the eigenvalues of the
    adjacency matrix and the weights of the GSR layer, and adding an L2 norm to the MSE loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from gan.model import Discriminator, gaussian_noise_layer
from set_seed import set_seed

set_seed(42)


def train(model, subjects_adj, subjects_labels, args, verbose=False):
    criterion = nn.MSELoss()
    # create a validation set
    train_g, val_G, train_labels, val_labels = train_test_split(
        subjects_adj, subjects_labels, test_size=0.2, random_state=42)

    bce_loss = nn.BCELoss()
    netD = Discriminator(args).to(args.device)
    if verbose:
        print(model)
        print(netD)
    optimizerG = optim.Adam(model.parameters(), lr=args.lr)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)

    best_model = None
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        with (torch.autograd.set_detect_anomaly(True)):
            model.train()
            netD.train()
            g_loss = []
            d_loss = []
            epoch_error = []
            for (lr_A, lr_X), hr in zip(train_g, train_labels):
                optimizerD.zero_grad()
                optimizerG.zero_grad()

                model_outputs = model(lr_A, lr_X)

                hr = torch.from_numpy(hr).type(torch.FloatTensor).to(args.device)

                mse_loss = criterion(model_outputs, hr) + torch.norm(model.layer.weights, 2)

                error = criterion(model_outputs, hr)
                real_data = model_outputs.detach()
                fake_data = gaussian_noise_layer(hr, args)

                d_real = netD(real_data)
                d_fake = netD(fake_data)

                dc_loss_real = bce_loss(d_real, torch.ones_like(d_real))
                dc_loss_fake = bce_loss(d_fake, torch.zeros_like(d_fake))
                dc_loss = dc_loss_real + dc_loss_fake

                dc_loss.backward()
                optimizerD.step()

                d_fake = netD(gaussian_noise_layer(hr, args))

                gen_loss = bce_loss(d_fake, torch.ones_like(d_fake))
                generator_loss = gen_loss + mse_loss
                generator_loss.backward()
                optimizerG.step()

                g_loss.append(generator_loss.item())
                epoch_error.append(error.item())
                d_loss.append(dc_loss.item())

            with torch.no_grad():
                val_error = []
                model.eval()
                netD.eval()
                for (lr_A, lr_X), hr in zip(val_G, val_labels):
                    hr = torch.from_numpy(hr).type(torch.FloatTensor).to(args.device)
                    model_outputs = model(lr_A, lr_X)
                    error = criterion(model_outputs, hr)
                    val_error.append(error.item())
            if verbose:
                print(
                    f'Epoch {epoch} - G Loss: {np.mean(g_loss)} - D Loss: {np.mean(d_loss)} - Error: {np.mean(epoch_error)}')
                print(f'Epoch {epoch} - Val Error: {np.mean(val_error)}')

            if np.mean(val_error) < best_loss:
                best_loss = np.mean(val_error)
                best_epoch = epoch
                best_model = model
            elif epoch - best_epoch > args.grace_period and epoch > args.min_epochs:
                break

    return best_model


def test(model, test_g, test_labels, args):
    loss_MAE = nn.L1Loss()
    test_error = []

    for (lr_A, lr_X), hr in zip(test_g, test_labels):
        all_zeros_hr = not np.any(hr)
        all_zeros_lr = not torch.any(lr_A)
        if not all_zeros_lr and not all_zeros_hr:
            hr = torch.from_numpy(hr).type(torch.FloatTensor).to(args.device)
            preds = model(lr_A, lr_X)
            error = loss_MAE(preds, hr)
            test_error.append(error.item())

    return np.mean(test_error)
