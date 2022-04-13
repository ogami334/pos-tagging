from typing import Dict, List

from pos_tagging.utils.registrable import Registrable


class Model(Registrable):
    def __init__(self, num_features: int, num_classes: int):
        self._num_features = num_features
        self.num_classes = num_classes

    def predict(self, word_features: List[List[int]]) -> List[int]:
        """
        Make prediction given word features of s single sentence.
        """
        raise NotImplementedError()

    def update(self, word_features: List[List[int]], tags: List[int]) -> Dict:
        """
        Update the model parameters using with a single training sentence.
        """
        raise NotImplementedError()

    def save(self, save_directory: str):
        raise NotImplementedError()

    @classmethod
    def load(cls, directory: str) -> "Model":
        raise NotImplementedError()
