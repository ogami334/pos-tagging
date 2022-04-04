from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_title")
class IsTitleExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return [f"IS_TITLE" if w.istitle() else None for w in words]
