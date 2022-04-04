from typing import List

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("context_word")
class ContextWordExtractor(FeatureExtractor):
    def __init__(self, context_distance: int):
        self.context_distance = context_distance
        self.max_distance = abs(self.context_distance)

    def get_features(self, words: List[str]) -> List[str]:
        features = []

        paddings = [f"PAD" for i in range(self.max_distance)]
        padded_words = paddings + words + paddings
        for i, word_position in enumerate(range(len(paddings), len(paddings) + len(words))):
            context_position = word_position + self.context_distance
            feature = f"{padded_words[context_position]}@{self.context_distance}"
            features.append(feature)
        return features
