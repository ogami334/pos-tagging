from typing import Dict

import numpy as np

from machine_learning.utils.registrable import Registrable


class Model(Registrable):
    def batch_predict(self, batch_features: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def update(self, batch_features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Parameters
        ----------
        batch_features : ``np.array`` (batch_size, max_feature_length)
        labels : ``np.array`` (batch_size,)
        """
        raise NotImplementedError()

    def save(self, save_directory: str):
        raise NotImplementedError()

    @classmethod
    def load(cls, directory: str) -> "Model":
        raise NotImplementedError()
