import logging
from typing import Dict, List

import tqdm

from pos_tagging.utils.registrable import FromConfig

from .models import Model

logger = logging.getLogger(__name__)




class Trainer(FromConfig):
    def __init__(self, num_epochs: int, patience: int):
        self.num_epochs = num_epochs
        self.patience = patience
        self.inv_dict = {0: 'NN', 1: 'IN', 2: 'NNP', 3: 'DT', 4: 'JJ', 5: 'NNS', 6: ',', 7: '.', 8: 'CD', 9: 'RB', 10: 'VBD', 11: 'VB', 12: 'CC', 13: 'TO', 14: 'VBZ', 15: 'VBN', 16: 'PRP', 17: 'VBG', 18: 'VBP', 19: 'MD', 20: 'POS', 21: 'PRP$', 22: '$', 23: '``', 24: "''", 25: ':', 26: 'WDT', 27: 'JJR', 28: 'RP', 29: 'NNPS', 30: 'WP', 31: 'WRB', 32: 'JJS', 33: 'RBR', 34: '-RRB-', 35: '-LRB-', 36: 'EX', 37: 'RBS', 38: 'PDT', 39: 'FW', 40: 'WP$', 41: '#', 42: 'UH', 43: 'SYM', 44: 'LS'}

    def _training_loop(self, model: Model, training_data: List[Dict]) -> float:
        model.set_train_mode()
        total_num_correct_predictions = 0
        num_prediction_points = sum(len(d["tag_ids"]) for d in training_data)
        for data in tqdm.tqdm(training_data):
            output_dict = model.update(data["feature_ids"], data["tag_ids"])
            total_num_correct_predictions += sum(i == j for i, j in zip(output_dict["prediction"], data["tag_ids"]))
        training_accuracy = total_num_correct_predictions / num_prediction_points
        return training_accuracy

    def _validation_loop(self, model: Model, validation_data: List[Dict]) -> float:
        model.set_eval_mode()
        num_correct_predictions = 0
        num_prediction_points = sum(len(d["tag_ids"]) for d in validation_data)
        for data in tqdm.tqdm(validation_data):
            model_prediction = model.predict(data["feature_ids"])
            for index, t in enumerate(zip(model_prediction, data["tag_ids"])):
                i, j = t
                if i!= j:
                    # print("word:{}".format(data["words"][index]))
                    # print(f"predicted:{self.inv_dict[i]}")
                    # print("label:{}".format(data["tags"][index]))
                    # print()
                    pass
                else:
                    num_correct_predictions += 1
            # num_correct_predictions += sum(i == j for i, j in zip(model_prediction, data["tag_ids"]))
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
