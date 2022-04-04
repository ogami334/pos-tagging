from abc import ABCMeta, abstractmethod
from typing import List


class PoSTagger(metaclass=ABCMeta):
    @abstractmethod
    def tag_words(self, words: List[str]) -> List[str]:
        raise NotImplementedError()
