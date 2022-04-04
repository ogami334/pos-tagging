from pos_tagging.feature_extractors.is_digit import IsDigitExtractor
from pos_tagging.feature_extractors.is_title import IsTitleExtractor
from pos_tagging.feature_extractors.is_upper import IsUpperExtractor


def test_is_digit_extractor():
    feature_extractor = IsDigitExtractor()
    assert feature_extractor.get_features(["This", "is", "000"]) == ["DIGIT:False", "DIGIT:False", "DIGIT:True"]


def test_is_title_extractor():
    feature_extractor = IsTitleExtractor()
    assert feature_extractor.get_features(["This", "is", "test"]) == ["TITLE:True", "TITLE:False", "TITLE:False"]


def test_is_upper_extractor():
    feature_extractor = IsUpperExtractor()
    assert feature_extractor.get_features(["This", "is", "TEST"]) == ["UPPER:False", "UPPER:False", "UPPER:True"]
