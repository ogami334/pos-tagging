from machine_learning.feature_extractors.context_word_extractor import (
    ContextWordExtractor,
)


def test_extract_center_word_features():
    feature_extractor = ContextWordExtractor(context_distance=0)
    assert feature_extractor.get_features(["This", "is", "test"]) == ["This@0", "is@0", "test@0"]


def test_extract_next_word_features():
    feature_extractor = ContextWordExtractor(context_distance=1)
    assert feature_extractor.get_features(["This", "is", "test"]) == ["is@1", "test@1", "PAD@1"]


def test_extract_previous_word_features():
    feature_extractor = ContextWordExtractor(context_distance=-1)
    assert feature_extractor.get_features(["This", "is", "test"]) == ["PAD@-1", "This@-1", "is@-1"]
