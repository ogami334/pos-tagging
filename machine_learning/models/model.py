from typing import Dict

import numpy as np

from machine_learning.utils.registrable import Registrable


class Model(Registrable):
    def predict(self, word_features: np.ndarray) -> np.ndarray:
        """
        Make prediction given word features of a sentence.

        Parameters
        ----------
        word_features : ``np.array`` (sentence_length, max_feature_length)
            The features of the words in a sentence.
        """

        raise NotImplementedError()

    def update(self, word_features: np.ndarray, tags: np.ndarray) -> Dict:
        """
        Update the model parameters using with a single training sentence.

        Parameters
        ----------
        word_features : ``np.array`` (batch_size, max_feature_length)
            The features of the words in a sentence.
        tags : ``np.array`` (batch_size,)
            The PoS tags of the words.
        """
        raise NotImplementedError()

    def save(self, save_directory: str):
        raise NotImplementedError()

    @classmethod
    def load(cls, directory: str) -> "Model":
        raise NotImplementedError()
