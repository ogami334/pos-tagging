from typing import List

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("context_word")
class ContextWordExtractor(FeatureExtractor):
    def __init__(self, context_distance: int, lower_words: bool = False):
        self.context_distance = context_distance
        self.max_distance = abs(self.context_distance)
        self.lower_words = lower_words

    def get_features(self, words: List[str]) -> List[str]:
        if self.lower_words:
            words = [w.lower() for w in words]

        features = []

        paddings = [f"PAD" for _ in range(self.max_distance)]
        padded_words = paddings + words + paddings
        for i, word_position in enumerate(range(len(paddings), len(paddings) + len(words))):
            context_position = word_position + self.context_distance
            feature = f"{padded_words[context_position]}@{self.context_distance}"
            features.append(feature)
        return features
