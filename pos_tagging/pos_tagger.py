import json
from pathlib import Path
from typing import List

from pos_tagging.feature_extractors import FeatureExtractor, extract_features

from .models import Model
from .vocabulary import Vocabulary


class PoSTagger:
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
        predictions = self.model.predict(feature_ids_list)
        predicted_tags = self.tag_vocabulary.get_tokens_from_indices(predictions)
        return predicted_tags

    @classmethod
    def load(cls, result_save_directory: str) -> "PoSTagger":
        result_save_directory = Path(result_save_directory)
        config = json.load(open(result_save_directory / "config.json", "r"))
        feature_vocabulary = Vocabulary.load(result_save_directory / "feature_vocabulary.txt")
        tag_vocabulary = Vocabulary.load(result_save_directory / "tag_vocabulary.txt")
        model = Model.by_name(config["model"]["type"]).load(result_save_directory)
        return PoSTagger(
            feature_extractors=[FeatureExtractor.from_config(c) for c in config["feature_extractors"]],
            feature_vocabulary=feature_vocabulary,
            tag_vocabulary=tag_vocabulary,
            model=model,
        )
