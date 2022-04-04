from typing import List

from .feature_extractor import FeatureExtractor


@FeatureExtractor.register("is_digit")
class IsDigitExtractor(FeatureExtractor):
    def get_features(self, words: List[str]) -> List[str]:
        return [f"DIGIT:{w.isdigit()}" for w in words]
