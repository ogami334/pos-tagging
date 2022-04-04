from typing import List

from machine_learning.feature_extractors import FeatureExtractor, extract_features
from pos_tagging.pos_tagger import PoSTagger

from .data_iterator import convert_feature_ids_list
from .models import Model
from .vocabulary import Vocabulary


class MLPoSTagger(PoSTagger):
    def __init__(
        self,
        feature_extractors: List[FeatureExtractor],
        feature_vocabulary: Vocabulary,
        tag_vocabulary: Vocabulary,
        model: Model,
    ):
        self.feature_extractors = feature_extractors
        self.feature_vocabulary = feature_vocabulary
        self.tag_vocabulary = tag_vocabulary
        self.model = model

    def tag_words(self, words: List[str]) -> List[str]:
        features_list = extract_features(words, self.feature_extractors)
        feature_ids_list = [self.feature_vocabulary.get_indices_from_tokens(features) for features in features_list]
        batch_features = convert_feature_ids_list(feature_ids_list)
        predictions = self.model.batch_predict(batch_features)
        predicted_tags = self.tag_vocabulary.get_tokens_from_indices(predictions)
        return predicted_tags
