"""
This file contains the initialization functions for the weights of the neural network layers.

It was adapted from the following source:
https://github.com/basiralab/AGSR-Net
"""

import numpy as np

from set_seed import set_seed

set_seed(42)


def weight_variable_glorot(output_dim, input_dim=None):
    if input_dim is None:
        input_dim = output_dim
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = np.random.uniform(-init_range, init_range,
                                (input_dim, output_dim))
    return initial
