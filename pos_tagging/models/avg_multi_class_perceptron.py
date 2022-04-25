from pathlib import Path
from typing import Dict, List

import numpy as np
import copy

from .model import Model

PARAMETER_FILE_NAME = "parameters.npz"


@Model.register("avg_multi_class_perceptron")
class AvgMultiClassPerceptron(Model):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__(num_features, num_classes)
        self.weights = np.zeros((num_features, num_classes)) # 学習重みをメモしておく場所
        self.bias = np.zeros(num_classes) # 学習時バイアスをメモしておく場所
        self.u_weights = np.zeros((num_features, num_classes)) #実行重み & 更新される重み
        self.u_bias = np.zeros(num_classes) #実行バイアス & 更新される重み
        self.tmp_weights = np.zeros((num_features, num_classes)) # 学習重みをメモしておく場所
        self.tmp_bias = np.zeros(num_classes) # 学習時バイアスをメモしておく場所
        self.t = 0

    def set_train_mode(self):
        super().set_train_mode()
        # self.weights += self.u_weights / (self.t + 1)
        # self.bias += self.u_bias / (self.t + 1)
        self.weights = self.tmp_weights.copy()
        self.bias = self.tmp_bias.copy()
    
    def set_eval_mode(self):
        super().set_eval_mode()
        self.tmp_weights = self.weights.copy()
        self.tmp_bias = self.bias.copy()
        self.weights -= self.u_weights / (self.t + 1)
        self.bias -= self.u_bias / (self.t + 1)
        # print(np.nonzero(self.u_weights))
        #新たに行列を作る操作が必要かも

    def predict(self, word_features: List[List[int]]) -> List[int]:
        predicted_tags = []
        for fs in word_features:
            scores = self.weights[fs].sum(axis=0) + self.bias
            # fsに1が立っているindexが入っていて、1の部分の重みを全て足す実装になっている
            predicted_tag = scores.argmax()
            predicted_tags.append(predicted_tag)
        return predicted_tags

    def update(self, word_features: List[List[int]], tags: List[int]) -> Dict:
        predicted_labels = self.predict(word_features)

        # incorrect_indices = [
        #     i for i, (ground_truth, prediction) in enumerate(zip(tags, predicted_labels)) if ground_truth != prediction
        # ]
        #データの数だけ足すような気がする
        for i, (ground_truth_tag_idx, prediction_tag_idx) in enumerate(zip(tags, predicted_labels)):
            self.t += 1
            if ground_truth_tag_idx == prediction_tag_idx:
                continue
            # ground_truth_tag_idx = tags[i]
            # prediction_tag_idx = predicted_labels[i]
            features = word_features[i]

            self.weights[:, ground_truth_tag_idx][features] += 1
            self.bias[ground_truth_tag_idx] += 1
            self.weights[:, prediction_tag_idx][features] -= 1
            self.bias[prediction_tag_idx] -= 1
            # print(self.t)
            self.u_weights[:, ground_truth_tag_idx][features] += self.t
            self.u_bias[ground_truth_tag_idx] += self.t
            self.u_weights[:, prediction_tag_idx][features] -= self.t
            self.u_bias[prediction_tag_idx] -= self.t
        # print(self.t)
        return {"prediction": predicted_labels}

    def save(self, save_directory: str):
        np.savez(Path(save_directory) / PARAMETER_FILE_NAME, weights=self.weights, bias=self.bias)

    @classmethod
    def load(cls, save_directory: str) -> "AvgMultiClassPerceptron":
        parameters = np.load(Path(save_directory) / PARAMETER_FILE_NAME)
        num_features, num_classes = parameters["weights"].shape
        model = AvgMultiClassPerceptron(num_features, num_classes)
        model.weights = parameters["weights"]
        model.bias = parameters["bias"]
        return model
