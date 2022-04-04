import random
from typing import Dict, Iterator, List

import numpy as np


def convert_feature_ids_list(feature_ids_list: List[List[int]], padding_index: int) -> np.ndarray:
    max_feature_length = max(len(features) for features in feature_ids_list)
    padded_feature_list = [
        features + [padding_index] * (max_feature_length - len(features)) for features in feature_ids_list
    ]
    return np.stack(padded_feature_list)


class DataIterator:
    def __init__(self, feature_field_name: str, label_field_name: str = None, padding_index: int = 0):
        self.feature_field_name = feature_field_name
        self.label_field_name = label_field_name

        self.padding_index = padding_index

    def make_batch(self, data_list: List[Dict]) -> Dict[str, np.ndarray]:
        feature_ids_list = [data[self.feature_field_name] for data in data_list]
        batch = {self.feature_field_name: convert_feature_ids_list(feature_ids_list, padding_index=self.padding_index)}

        if self.label_field_name is not None:
            label_list = [data[self.label_field_name] for data in data_list]
            batch[self.label_field_name] = np.array(label_list)
        return batch

    def batch_iterate(self, data_list: List[Dict], batch_size: int, shuffle: bool) -> Iterator[Dict[str, np.ndarray]]:
        if shuffle:
            random.shuffle(data_list)
        num_data = len(data_list)
        for start_index in range(0, num_data, batch_size):
            end_index = min(start_index + batch_size, num_data)
            yield self.make_batch(data_list[start_index:end_index])
