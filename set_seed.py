"""
Helper function to set seed for reproducibility. Even though we use it in every file, it seems like there is some
variability in the results. This might be because of the inherent randomness of some operations on the GPU, We
apologize for the inconvenience.
"""

import random

import numpy as np
import torch


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.empty_cache()
    # Check for CUDA (GPU support) and set device accordingly
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # For multi-GPU setups
        # Additional settings for ensuring reproducibility on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
