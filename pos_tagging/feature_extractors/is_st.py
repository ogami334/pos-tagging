from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_st")
class IsUpperExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return ["IS_ST" if (len(w) >= 2) and (w[-2:] =="st") else None for w in words]
