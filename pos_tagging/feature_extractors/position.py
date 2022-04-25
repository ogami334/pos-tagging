from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("position")
class IsUpperExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return [f"POS:{index}" for index, w in enumerate(words)]
