from pathlib import Path
from typing import Dict

import numpy as np

from .model import Model

PARAMETER_FILE_NAME = "parameters.npz"


@Model.register("multi_class_perceptron")
class MultiClassPerceptron(Model):
    def __init__(self, num_features: int, num_classes: int):
        self._num_features = num_features
        self.weights = np.zeros((num_features, num_classes))
        self.bias = np.zeros(num_classes)

    def batch_predict(self, batch_features: np.ndarray):
        scores = self.weights[batch_features].sum(axis=1) + self.bias
        predicted_labels = scores.argmax(axis=1)
        return predicted_labels

    def update(self, batch_features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Parameters
        ----------
        batch_features : ``np.array`` (batch_size, max_feature_length)
        labels : ``np.array`` (batch_size,)

        Returns
        -------
        num_incorrect_predictions : float
        """
        predicted_labels = self.batch_predict(batch_features)

        incorrect_indices = (labels != predicted_labels).nonzero()[0]

        for feature, to_add_idx, to_sub_idx in zip(
            batch_features[incorrect_indices], labels[incorrect_indices], predicted_labels[incorrect_indices]
        ):
            nonzero_feature = feature[feature.nonzero()]  # remove zero because it is padding

            self.weights[:, to_add_idx][nonzero_feature] += 1
            self.bias[to_add_idx] += 1
            self.weights[:, to_sub_idx][nonzero_feature] -= 1
            self.bias[to_sub_idx] -= 1

        return {"prediction": predicted_labels}

    def save(self, save_directory: str):
        np.savez(Path(save_directory) / PARAMETER_FILE_NAME, weights=self.weights, bias=self.bias)

    @classmethod
    def load(cls, save_directory: str) -> "MultiClassPerceptron":
        parameters = np.load(Path(save_directory) / PARAMETER_FILE_NAME)
        num_features, num_classes = parameters["weights"].shape()
        model = MultiClassPerceptron(num_features, num_classes)
        model.weights = parameters["weights"]
        model.bias = parameters["bias"]
        return model
