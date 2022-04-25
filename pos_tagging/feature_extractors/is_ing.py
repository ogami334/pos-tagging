from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_ing")
class IsUpperExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return ["IS_ING" if (len(w) >= 3) and (w[-3:] =="ing") else None for w in words]
