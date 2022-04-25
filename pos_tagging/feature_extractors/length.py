from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("length")
class IsUpperExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return [f"LEN:{len(w)}" for w in words]
