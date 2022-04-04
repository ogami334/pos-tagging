from typing import List, Optional

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_digit")
class IsDigitExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[Optional[str]]:
        return [f"IS_DIGIT" if w.isdigit() else None for w in words]
