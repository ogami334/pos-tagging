from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_ed")
class IsUpperExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return ["IS_ED" if (len(w) >= 2) and (w[-2:] =="ed") else None for w in words]
