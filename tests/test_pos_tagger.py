import json
from collections import Counter
from pathlib import Path
from typing import List

from pos_tagging.feature_extractors import FeatureExtractor, extract_features
from pos_tagging.feature_extractors.context_word_extractor import ContextWordExtractor
from pos_tagging.feature_extractors.is_title import IsTitleExtractor
from pos_tagging.models import Model
from pos_tagging.models.multi_class_perceptron import MultiClassPerceptron
from pos_tagging.vocabulary import Vocabulary


def test_save_and_load():

    data = [["This", "is", "test", "."], ["This", "is", "another", "test", "."]]

    counter = Counter()
    for words in data:
        counter.update(words)
    feature_vocabulary = Vocabulary.build_from_counter(counter)
    tag_vocabulary = Vocabulary.build_from_counter(Counter({"TEST": 1, "TEST2": 1}))

    model = MultiClassPerceptron(num_features=len(feature_vocabulary), num_classes=len(tag_vocabulary))

    feature_extractors = [
        ContextWordExtractor(0),
        ContextWordExtractor(-1),
        ContextWordExtractor(1),
        IsTitleExtractor(),
    ]
    feature_vocabulary = Vocabulary()
