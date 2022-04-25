from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("first_ch")
class IsUpperExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return [f"{w[0]}:first" for w in words]
