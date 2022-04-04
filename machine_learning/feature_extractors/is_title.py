from typing import List

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_title")
class IsTitleExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[str]:
        return [f"TITLE:{w.istitle()}" for w in words]
