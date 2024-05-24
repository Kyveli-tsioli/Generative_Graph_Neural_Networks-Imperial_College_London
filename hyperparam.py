import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import wandb
from gan.config import Args
from gan.model import GUS
from gan.preprocessing import degree_normalisation, preprocess_data
from gan.train import test as test_model
from gan.train import train as train_model
from set_seed import set_seed
from utils import load_csv_files

set_seed(42)

WANDB_API_KEY = "..."  # Replace with your own API key


def hyperparameter_search(lr_train, lr_test, hr_train, hr_test, device):
    """
    A function to perform hyperparameter search using Weights and Biases
    """
    wandb.login(key=WANDB_API_KEY)

    args = Args()
    args.device = device
    args.normalisation_function = degree_normalisation

    sweep_config = {
        "name": "GUS-GAN",
        "method": "random",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "lr": {
                "values": [0.0001, 0.0005, 0.001, 0.005, 0.01]
            },
            "init_x_method": {
                "values": ["eye", "topology"]
            },
            "hidden_dim": {
                "values": [128, 256, 512, 1024]
            },
            "k": {
                "values": [2, 3, 4, 5, 7]
            },
            "alpha": {
                "values": [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="dgbl", entity="live-love-graph")

    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            args.lr = config["lr"]
            args.hidden_dim = config["hidden_dim"]
            args.init_x_method = config["init_x_method"]
            args.k = config["k"]
            args.alpha = config["alpha"]

            model = GUS(args.ks, args).to(device)
            model = train_model(model, lr_train, hr_train, args)
            scores = test_model(model, lr_test, hr_test, args)
            wandb.log({"loss": scores})

    wandb.agent(sweep_id, train, count=50)

    wandb.finish()


if __name__ == "__main__":
    """
    The main function to run the hyperparameter search. Load and preprocess the data, and then run the hyperparameter 
    search.
    
    To run this file, you will need to replace the WANDB_API_KEY with your own API key from Weights and Biases.
    We do not recommend running this file locally, as it will take a long time to complete. Instead, we recommend 
    using the university GPU cluster.
    """

    # Set a fixed random seed for reproducibility across multiple libraries
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    torch.cuda.empty_cache()

    # Check for CUDA (GPU support) and set device accordingly
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # For multi-GPU setups
        # Additional settings for ensuring reproducibility on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    # Implementation
    lr_train_data, hr_train_data, lr_test_data = load_csv_files(return_matrix=True)

    args = Args()
    args.device = device
    args.normalisation_function = degree_normalisation

    lr_train_A, lr_train_X = preprocess_data(lr_train_data, args)
    lr_train_data = torch.stack([lr_train_A, lr_train_X], dim=1)

    lr_train, lr_test, hr_train, hr_test = train_test_split(lr_train_data, hr_train_data, test_size=0.2,
                                                            random_state=42)

    hyperparameter_search(lr_train, lr_test, hr_train, hr_test, device)
