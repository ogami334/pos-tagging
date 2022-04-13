from pathlib import Path
from typing import Dict, List

import numpy as np

from .model import Model

PARAMETER_FILE_NAME = "parameters.npz"


@Model.register("multi_class_perceptron")
class MultiClassPerceptron(Model):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__(num_features, num_classes)
        self.weights = np.zeros((num_features, num_classes))
        self.bias = np.zeros(num_classes)

    def predict(self, word_features: List[List[int]]) -> List[int]:
        predicted_tags = []
        for fs in word_features:
            scores = self.weights[fs].sum(axis=0) + self.bias
            predicted_tag = scores.argmax()
            predicted_tags.append(predicted_tag)
        return predicted_tags

    def update(self, word_features: List[List[int]], tags: List[int]) -> Dict:
        predicted_labels = self.predict(word_features)

        incorrect_indices = [
            i for i, (ground_truth, prediction) in enumerate(zip(tags, predicted_labels)) if ground_truth != prediction
        ]

        for i in incorrect_indices:
            ground_truth_tag_idx = tags[i]
            prediction_tag_idx = predicted_labels[i]
            features = word_features[i]

            self.weights[:, ground_truth_tag_idx][features] += 1
            self.bias[ground_truth_tag_idx] += 1
            self.weights[:, prediction_tag_idx][features] -= 1
            self.bias[prediction_tag_idx] -= 1

        return {"prediction": predicted_labels}

    def save(self, save_directory: str):
        np.savez(Path(save_directory) / PARAMETER_FILE_NAME, weights=self.weights, bias=self.bias)

    @classmethod
    def load(cls, save_directory: str) -> "MultiClassPerceptron":
        parameters = np.load(Path(save_directory) / PARAMETER_FILE_NAME)
        num_features, num_classes = parameters["weights"].shape
        model = MultiClassPerceptron(num_features, num_classes)
        model.weights = parameters["weights"]
        model.bias = parameters["bias"]
        return model
