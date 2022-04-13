from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_upper")
class IsUpperExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return ["IS_UPPER" if w.isupper() else None for w in words]
