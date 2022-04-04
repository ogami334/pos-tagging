import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

import click

from pos_tagging.feature_extractors import FeatureExtractor, extract_features
from pos_tagging.models import Model
from pos_tagging.read_wsj_data import parse_wsj_file
from pos_tagging.trainer import Trainer
from pos_tagging.utils.import_util import import_submodules
from pos_tagging.vocabulary import Vocabulary

logger = logging.getLogger(__name__)
fmt = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
logging.basicConfig(level=logging.INFO, format=fmt)


def read_and_extract_features(file_path: str, feature_extractors: List[FeatureExtractor]):
    data_list = [data for data in parse_wsj_file(file_path)]
    for data in data_list:
        data["features"] = extract_features(data["words"], feature_extractors)
    return data_list


def add_ids_to_data(data_list: List[Dict], feature_vocabulary: Vocabulary, tag_vocabulary: Vocabulary) -> List[Dict]:
    # convert raw data into indices
    for data in data_list:
        data["feature_ids"] = [feature_vocabulary.get_indices_from_tokens(features) for features in data["features"]]
        data["tag_ids"] = tag_vocabulary.get_indices_from_tokens(data["tags"])
    return data_list


@click.command()
@click.argument("config-path", type=click.Path(exists=True))
@click.argument("result-save-directory", type=click.Path(exists=False))
def train_multi_class_perceptron(config_path: str, result_save_directory: str):
    import_submodules("pos_tagging")

    # preparing result_save_directory
    result_save_directory = Path(result_save_directory)
    result_save_directory.mkdir(exist_ok=True, parents=True)

    file_handler = logging.FileHandler(result_save_directory / "out.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    config = json.load(open(config_path))

    with open(result_save_directory / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # extract features from FeatureExtractor
    feature_extractors = [FeatureExtractor.from_config(f) for f in config["feature_extractors"]]
    train_data = read_and_extract_features(config["train_data_path"], feature_extractors)

    # build vocabulary
    label_counter = Counter()
    feature_counter = Counter()
    for data in train_data:
        label_counter.update(data["tags"])
        for fs in data["features"]:
            feature_counter.update(fs)
    tag_vocabulary = Vocabulary.build_from_counter(label_counter)
    feature_vocabulary = Vocabulary.build_from_counter(
        feature_counter, use_padding=True, use_unknown=True, **config["feature_vocabulary"]
    )
    logger.info(f"Feature vocabulary size: {len(feature_vocabulary)}")
    logger.info(f"Tag vocabulary size: {len(tag_vocabulary)}")

    feature_vocabulary.save(result_save_directory / "feature_vocabulary.txt")
    tag_vocabulary.save(result_save_directory / "tag_vocabulary.txt")

    # convert raw data into indices
    train_data = add_ids_to_data(train_data, feature_vocabulary, tag_vocabulary)
    validation_data = read_and_extract_features(config["validation_data_path"], feature_extractors)
    validation_data = add_ids_to_data(validation_data, feature_vocabulary, tag_vocabulary)

    model = Model.from_config(config["model"], num_features=len(feature_vocabulary), num_classes=len(tag_vocabulary))
    trainer = Trainer.from_config(config["trainer"])
    metrics = trainer.train(model, train_data, validation_data, result_save_directory)
    with open(Path(result_save_directory) / "metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    train_multi_class_perceptron()
