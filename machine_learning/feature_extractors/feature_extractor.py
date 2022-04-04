from abc import abstractmethod
from typing import List, Optional

from machine_learning.utils.registrable import Registrable


class FeatureExtractor(Registrable):
    @abstractmethod
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        raise NotImplementedError()


def extract_features(words: List[str], feature_extractors: List[FeatureExtractor]) -> List[List[str]]:
    features = [[] for _ in range(len(words))]

    for feature_extractor in feature_extractors:
        for i, word_feature in enumerate(feature_extractor.get_features(words)):
            if word_feature is not None:
                features[i].append(word_feature)
    return features
