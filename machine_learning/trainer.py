import logging
from typing import Dict, List

import tqdm

from machine_learning.utils.registrable import FromConfig

from .models import Model

logger = logging.getLogger(__name__)


class Trainer(FromConfig):
    def __init__(self, num_epochs: int, patience: int):
        self.num_epochs = num_epochs
        self.patience = patience

    def _training_loop(self, model: Model, training_data: List[Dict]) -> float:
        total_num_correct_predictions = 0
        num_prediction_points = sum(len(d["tag_ids"]) for d in training_data)
        for data in tqdm.tqdm(training_data):
            output_dict = model.update(data["feature_ids"], data["tag_ids"])
            total_num_correct_predictions += sum(i == j for i, j in zip(output_dict["prediction"], data["tag_ids"]))
        training_accuracy = total_num_correct_predictions / num_prediction_points
        return training_accuracy

    def _validation_loop(self, model: Model, validation_data: List[Dict]) -> float:
        num_correct_predictions = 0
        num_prediction_points = sum(len(d["tag_ids"]) for d in validation_data)
        for data in tqdm.tqdm(validation_data):
            model_prediction = model.predict(data["feature_ids"])
            num_correct_predictions += sum(i == j for i, j in zip(model_prediction, data["tag_ids"]))
        validation_accuracy = num_correct_predictions / num_prediction_points
        return validation_accuracy

    def train(
        self,
        model: Model,
        training_data: List[Dict],
        validation_data: List[Dict],
        result_save_directory: str,
    ) -> Dict:

        best_validation_accuracy = 0.0
        metrics = {"best_epochs": 0, "best_validation_accuracy": best_validation_accuracy}
        num_epochs_with_no_best_validation = 0
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch}")

            logger.info(f"Training loop")
            training_accuracy = self._training_loop(model, training_data)
            logger.info(f"Training accuracy: {training_accuracy}")

            logger.info(f"Validation")
            validation_accuracy = self._validation_loop(model, validation_data)
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
