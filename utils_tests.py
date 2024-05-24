"""
This file contains unit tests for the functions in utils.py. The functions are tested for correctness and expected
behaviour.
"""

import unittest

import numpy as np
import torch

from utils import (
    load_csv_files,
    save_csv_prediction,
    three_fold_cross_validation,
    plot_evaluations,
)


class TestLoadCsvFiles(unittest.TestCase):
    lr_train_data, hr_train_data, lr_test_data = load_csv_files()
    lr_train_matrix, hr_train_matrix, lr_test_matrix = load_csv_files(
        return_matrix=True
    )

    def test_preprocessing(self):
        self.assertFalse(np.isnan(self.lr_train_data).any())
        self.assertFalse(np.isnan(self.hr_train_data).any())
        self.assertFalse(np.isnan(self.lr_test_data).any())
        self.assertFalse(np.isnan(self.lr_train_matrix).any())
        self.assertFalse(np.isnan(self.hr_train_matrix).any())
        self.assertFalse(np.isnan(self.lr_test_matrix).any())

    def test_preprocessing(self):
        self.assertFalse((self.lr_train_data < 0).any())
        self.assertFalse((self.hr_train_data < 0).any())
        self.assertFalse((self.lr_test_data < 0).any())
        self.assertFalse((self.lr_train_matrix < 0).any())
        self.assertFalse((self.hr_train_matrix < 0).any())
        self.assertFalse((self.lr_test_matrix < 0).any())

    def test_shapes_vector(self):
        self.assertEqual(self.lr_train_data.shape, (167, 12720))
        self.assertEqual(self.hr_train_data.shape, (167, 35778))
        self.assertEqual(self.lr_test_data.shape, (112, 12720))

    def test_shapes_matrix(self):
        self.assertEqual(self.lr_train_matrix.shape, (167, 160, 160))
        self.assertEqual(self.hr_train_matrix.shape, (167, 268, 268))
        self.assertEqual(self.lr_test_matrix.shape, (112, 160, 160))


class TestSaveCsvFile(unittest.TestCase):
    FILE_PATH = "submissions/test_submission.csv"

    def test_csv(self):
        predictions = np.random.rand(112, 35778)
        save_csv_prediction(predictions, file_name="test_submission")
        header = np.genfromtxt(self.FILE_PATH, delimiter=",", dtype=str, max_rows=1)
        submission = np.genfromtxt(self.FILE_PATH, delimiter=",", skip_header=1)

        self.assertEqual(header[0], "ID")
        self.assertEqual(header[1], "Predicted")

        for i, row in enumerate(submission[:10]):
            self.assertEqual(int(row[0]), i + 1)
            self.assertAlmostEqual(row[1], predictions[0][i], delta=1e-5)

        self.assertEqual(int(submission[-1][0]), 4007136)


class Test3CV(unittest.TestCase):
    class DullModel:
        def fit(self, X, Y, verbose=False):
            pass

        def predict(self, X):
            return torch.rand(2, 35778)

    model = DullModel()

    def test_3cv(self):
        lr_train_data = torch.rand(6, 12720)
        hr_train_data = torch.rand(6, 35778)

        lr_train_data[lr_train_data < 0] = 0
        hr_train_data[hr_train_data < 0] = 0

        model_init = lambda: self.model

        scores = three_fold_cross_validation(
            model_init, lr_train_data, hr_train_data, verbose=True, prediction_vector=True
        )

        assert len(scores) == 3


class TestMeasuresPlot(unittest.TestCase):

    def test_plot(self):
        scores = np.random.rand(3, 6)
        plot_evaluations(scores)


if __name__ == "__main__":
    unittest.main()
