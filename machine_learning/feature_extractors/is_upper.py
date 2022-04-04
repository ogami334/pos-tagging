from typing import List

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_upper")
class IsUpperExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[str]:
        return [f"UPPER:{w.isupper()}" for w in words]
