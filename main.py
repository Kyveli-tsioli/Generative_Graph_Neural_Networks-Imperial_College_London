"""
Main Script for Data Preprocessing, Model Training, and Prediction CSV Generation

This script includes a basic workflow for loading, preprocessing, 
training a model, and generating predictions in a machine learning task.

"""
# Imports
import torch
from sklearn.model_selection import train_test_split

from gan.config import Args
from gan.model import GUS
from gan.preprocessing import degree_normalisation, preprocess_data
from gan.train import train as train_model
from utils import load_csv_files, three_fold_cross_validation, save_csv_prediction, plot_evaluations
from set_seed import set_seed


# Configurations
# Set a fixed random seed for reproducibility across multiple libraries
random_seed = 42
set_seed(random_seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

# preprocessing of the data
lr_train_data, hr_train_data, lr_test_data = load_csv_files(return_matrix=True)

args = Args()
args.device = device
args.normalisation_function = degree_normalisation

# print the args
print(args)

lr_train_A, lr_train_X = preprocess_data(lr_train_data, args)
lr_train_data = torch.stack([lr_train_A, lr_train_X], dim=1)

lr_train, lr_test, hr_train, hr_test = train_test_split(lr_train_data, hr_train_data, test_size=0.2, random_state=42)


def model_init():
    model = GUS(args.ks, args).to(device)
    return model


# run the 3-fold cross-validation
cv_scores = three_fold_cross_validation(model_init, lr_train_data, hr_train_data, random_state=random_seed,
                                        verbose=True, prediction_vector=False, label_vector=False)

print(f"The average over the 3 folds is: {torch.mean(torch.tensor(cv_scores), dim=0)}")

# Train the model on the whole dataset
model = GUS(args.ks, args).to(device)
model = train_model(model, lr_train_data, hr_train_data, args=args, verbose=True)  # train on the whole dataset
# save the trained model
torch.save(model, "model.pth")

# Test the model on the test set
lr_test_A, lr_test_X = preprocess_data(lr_test_data, args)
lr_test_data = torch.stack([lr_test_A, lr_test_X], dim=1)

lr_test_predictions = model.predict(lr_test_data)
save_csv_prediction(lr_test_predictions, logs=True, pred_matrix=True)
plot_evaluations(cv_scores)
