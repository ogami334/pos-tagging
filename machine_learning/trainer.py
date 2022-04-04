import logging
from typing import Dict, List

import numpy as np
import tqdm

from machine_learning.utils.registrable import FromConfig

from .data_iterator import DataIterator
from .models import Model

logger = logging.getLogger(__name__)


class Trainer(FromConfig):
    def __init__(self, batch_size: int, num_epochs: int, patience: int):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience

    def _training_loop(
        self, model: Model, training_instances: List[Dict[str, np.ndarray]], data_iterator: DataIterator
    ) -> float:
        total_num_correct_predictions = 0
        for batch in tqdm.tqdm(
            data_iterator.batch_iterate(training_instances, batch_size=self.batch_size, shuffle=True)
        ):
            output_dict = model.update(batch["feature_ids"], batch["tag_ids"])
            total_num_correct_predictions += (output_dict["prediction"] == batch["tag_ids"]).sum()
        training_accuracy = total_num_correct_predictions / len(training_instances)
        return training_accuracy

    def _validation_loop(
        self, model: Model, validation_instances: List[Dict[str, np.ndarray]], data_iterator: DataIterator
    ) -> float:
        num_correct_predictions = 0
        for batch in tqdm.tqdm(
            data_iterator.batch_iterate(validation_instances, batch_size=self.batch_size, shuffle=False)
        ):
            model_prediction = model.batch_predict(batch["feature_ids"])

            num_correct_predictions += (model_prediction == batch["tag_ids"]).sum()
        validation_accuracy = num_correct_predictions / len(validation_instances)
        return validation_accuracy

    def train(
        self,
        model: Model,
        training_instances: List[Dict[str, np.ndarray]],
        validation_instances: List[Dict[str, np.ndarray]],
        data_iterator: DataIterator,
        result_save_directory: str,
    ) -> Dict:

        best_validation_accuracy = 0.0
        metrics = {"best_epochs": 0, "best_validation_accuracy": best_validation_accuracy}
        num_epochs_with_no_best_validation = 0
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch}")

            logger.info(f"Training loop")
            training_accuracy = self._training_loop(model, training_instances, data_iterator)
            logger.info(f"Training accuracy: {training_accuracy}")

            logger.info(f"Validation")
            validation_accuracy = self._validation_loop(model, validation_instances, data_iterator)
            logger.info(f"Validation accuracy: {validation_accuracy}")

            if best_validation_accuracy < validation_accuracy:
                num_epochs_with_no_best_validation = 0
                logger.info(f"Best validation accuracy so far. Save the model parameters...")
                best_validation_accuracy = validation_accuracy
                model.save(result_save_directory)

                metrics.update({"best_epoch": epoch, "best_validation_accuracy": best_validation_accuracy})
            else:
                num_epochs_with_no_best_validation += 1

            if num_epochs_with_no_best_validation == self.patience:
                logger.info("Run out of patience. Stop training.")

        return metrics
